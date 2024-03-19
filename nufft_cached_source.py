import numpy as np
import aspire
from aspire.source.image import ArrayImageSource
from aspire.nufft import Plan 
from aspire.numeric import fft, xp
from aspire.volume import rotated_grids
from aspire.utils import complex_type
import pycuda.driver as cuda


class NUFFTCachedSource(ArrayImageSource):
    
    def __init__(self,im,metadata = None,angles = None,batch_size = 1,stack_size = 1):
        super().__init__(im,metadata,angles)
        self.batch_size = batch_size
        self.stack_size = stack_size
        self.init_nufft_plans()

    def init_nufft_plans(self,epsilon = 1e-8):
        #TODO : figure out if its allowed to remove the conversion of fourier_pts to double in cufinufft to save up on memory.
        #TODO : figure out if batch size > 1 is usefull in backprojection to save up on memory
        #TODO : use different cufinufft options (gpu_method,gpu_device_id,gpu_stream)
        #cuda.init()
        #context = cuda.Device(6).make_context()
        forward_sz = (self.L, ) * 3
        num_plans = int(np.ceil(self.n/self.batch_size))
        forward_plans = [0 for i in range(num_plans)]
        for i in range(int(self.n / self.batch_size)):
            rots = self.rotations[i*self.batch_size : ((i+1) * self.batch_size)]
            pts_rot = rotated_grids(self.L,rots)
            pts_rot = pts_rot.reshape((3,self.batch_size * self.L ** 2))
            
            plan = Plan(sz=forward_sz,fourier_pts = pts_rot,ntransforms = self.stack_size,epsilon = epsilon)#,gpu_device_id=6)
            
            forward_plans[i] = plan

        #context.pop()
        self.forward_plans = forward_plans

    def vol_forward(self,vols,image_ind,image_num):
        #TODO : cache normal fft plan for ifft (if needed)
        #TODO : check _apply_source_filters method to see if there is more efficient usage
        #TODO : do not perform computations for images that are in the batch but arent requested
        batch_ind_start = int(image_ind / self.batch_size)
        batch_ind_stop = int(np.ceil((image_ind + image_num) / self.batch_size))
        num_batches = batch_ind_stop - batch_ind_start
        
        vols_proj = np.empty((self.stack_size,num_batches * self.batch_size,self.L,self.L),dtype= vols.dtype)
        
        if(vols.dtype != complex_type(vols.dtype)):
            vols = vols.astype(complex_type(vols.dtype))

        for i in range(0,num_batches):
            
            im_f = self.forward_plans[i + batch_ind_start].transform(vols) / self.L
            im_f = im_f.reshape(-1,self.L,self.L)

            if(self.L % 2 == 0):
                im_f[:,0,:] = 0
                im_f[:,:,0] = 0

            im_f = np.real(xp.asnumpy(fft.centered_ifft2(xp.asarray(im_f))))
            vols_proj[:,i*self.batch_size : ((i+1) * self.batch_size),:,:] = im_f.reshape(self.stack_size,self.batch_size,self.L,self.L)

        vols_proj = vols_proj[:,image_ind % self.batch_size : (image_ind % self.batch_size + image_num)] #Get only the requested images out of the computed batches
        vols_proj = aspire.image.Image(vols_proj).stack_reshape(-1)
        vols_proj = self._apply_source_filters(vols_proj,np.tile(np.arange(image_ind,image_ind + image_num),self.stack_size))
        vols_proj = vols_proj.stack_reshape((self.stack_size,image_num))
        return vols_proj


    def im_backward(self,ims,image_ind):
        
        image_num = ims.stack_shape[1]
        batch_ind_start = int(image_ind / self.batch_size)
        batch_ind_stop = int(np.ceil((image_ind + image_num) / self.batch_size))
        num_batches = batch_ind_stop - batch_ind_start

        ims_backproj = np.empty((self.stack_size,image_num,self.L,self.L,self.L),dtype= ims.dtype)

        ims = ims.stack_reshape(-1)
        ims = self._apply_source_filters(ims,np.tile(np.arange(image_ind,image_ind + image_num),self.stack_size))
        im_f = xp.asnumpy(fft.centered_fft2(xp.asarray(ims._data))) / (self.L**2)
        

        if(self.L % 2 == 0):
            im_f[:,0,:] = 0
            im_f[:,:,0] = 0

        im_f = im_f.reshape((self.stack_size,image_num,-1))
         
        for i in range(num_batches):
            ims_backproj[:,i] = np.real(self.forward_plans[i + batch_ind_start].adjoint(im_f[:,i])) / self.L
            #im_f[:,i:i+2].reshape((self.stack_size,-1))
        
        return ims_backproj
    
if __name__ == "__main__":
    #%%
    import aspire
    import numpy as np
    import pandas as pd
    from aspire.operators import RadialCTFFilter
    from aspire.source.simulation import Simulation
    from aspire.volume import LegacyVolume, Volume
    from utils import volsCovarEigenvec
    import time
    from covar_estimation import im_stack_backward
    # Specify parameters
    img_size = 15  # image size in square
    num_imgs = 200  # number of images
    dtype = np.float32

    
    c = 5
    vols = LegacyVolume(
        L=img_size,
        C=c,
        dtype=dtype,
    ).generate()

    sim = Simulation(
        unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
        n=num_imgs,
        vols=vols,
        dtype=dtype,
        amplitudes=1,
        offsets = 0
    )
    sim.n
    from nufft_cached_source import NUFFTCachedSource

    batch_size = 1
    src = NUFFTCachedSource(sim.images[:],pd.DataFrame(sim.get_metadata()),sim.angles,batch_size = batch_size,stack_size = c)
    src.filter_indices = sim.filter_indices
    src.unique_filters = sim.unique_filters
    
    im_num = 100
    im_ind = 7
    
    start = time.process_time()
    images_from_cached = src.vol_forward(vols,im_ind,im_num)
    im_backproj_from_cached = src.im_backward(images_from_cached,im_ind)
    print(time.process_time() - start)
    
    images = aspire.image.Image(np.zeros(images_from_cached.shape))
    
    start = time.process_time()
    for i in range(images.shape[0]):
        images[i] = sim.vol_forward(vols[i], im_ind, im_num)
    im_backproj = im_stack_backward(images,sim,im_ind)
    print(time.process_time() - start)
    '''
    x = aspire.image.Image(images.asnumpy()/images_from_cached.asnumpy())
    
    images[0].show()
    images_from_cached[0].show()
    '''
    err = np.linalg.norm(images - images_from_cached) / np.linalg.norm(vols)
    print(err)
    
    err_back = np.linalg.norm(im_backproj - im_backproj_from_cached) / np.linalg.norm(vols)
    print(err_back)

# %%