import pickle
import torch
from cryodrgn.source import ImageSource as CryoDRGNImageSource
from cryodrgn.ctf import load_ctf_for_training,compute_ctf
from cov3d.poses import get_phase_shift_grid,pose_cryoDRGN2APIRE
from cov3d.projection_funcs import centered_fft2,centered_ifft2

class ImageSource:
    def __init__(self,particles_path,ctf_path,poses_path,apply_preprocessing=True):
        self.image_source = CryoDRGNImageSource.from_file(particles_path)
        self.ctf_params = torch.tensor(load_ctf_for_training(self.resolution,ctf_path))
        self.freq_lattice = (torch.stack(get_phase_shift_grid(self.resolution),dim=0)/torch.pi/2).permute(1,2,0).reshape(self.resolution**2,2)

        with open(poses_path,'rb') as f:
            poses = pickle.load(f)
        rots,offsets = pose_cryoDRGN2APIRE(poses,self.resolution)
        self.rotations = torch.tensor(rots)
        self.offsets = torch.tensor(offsets)
        self.apply_preprocessing = apply_preprocessing

        self.whitening_filter = None
        if self.apply_preprocessing:
            self._preprocess_images()
        
    @property
    def resolution(self):
        return self.image_source.D

    def __len__(self):
        return self.image_source.n

    def get_ctf(self,index):
        ctf_params = self.ctf_params[index]
        freq_lattice = self.freq_lattice / ctf_params[:,0].view(-1,1,1)
        ctf = compute_ctf(freq_lattice,*torch.split(ctf_params[:,1:],1,1)).reshape(-1,self.resolution,self.resolution)
        
        ctf = ctf if not self.apply_preprocessing else ctf * self.whitening_filter

    def images(self,index,fourier=False):
        images = self.image_source.images(index)
        if(not self.apply_preprocessing and not fourier):
            return images

        images = centered_fft2(images)

        if self.apply_preprocessing:
            images *= self.whitening_filter

        if not fourier:
            images = centered_ifft2(images)

    
        return images


    def __getitem__(self,index):
        return self.images(index), self.get_ctf(index), self.rotations[index], self.offsets[index]


    def _preprocess_images(self,batch_size=1024):
        """
        Whitens images by estimating the noise PSD and apply it as a filter on all images.
        Additionally each image is normalized indivudally to have N(0,1) background noise.
        Implementation is based on ASPIRE:
            https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/noise/noise.py#L333
            https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/image/image.py#L27

        """
        mask = (torch.norm(self.freq_lattice,dim=1) >= 0.5).reshape(self.resolution,self.resolution)
        n = len(self)
        mean_est = 0
        noise_psd_est = torch.zeros((self.resolution,)*2)
        for i in range(0,n,batch_size):
            idx = torch.arange(i,min(i+batch_size,n))
            images = self.image_source.images(idx) * mask

            mean_est += torch.sum(images)
            noise_psd_est += torch.sum(torch.abs(centered_fft2(images))**2,dim=0)


        mean_est /= torch.sum(mask) * n
        noise_psd_est /= torch.sum(mask) * n

        noise_psd_est[self.resolution//2,self.resolution//2] -= mean_est ** 2


        self.whitening_filter = (1/torch.sqrt(noise_psd_est)).unsqueeze(0)

            
