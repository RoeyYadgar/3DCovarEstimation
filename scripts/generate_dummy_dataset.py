import numpy as np
import pandas as pd
import os
import shutil
from aspire.volume import Volume,LegacyVolume
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter,ArrayFilter
from aspire.noise import WhiteNoiseAdder
import starfile
import click

@click.command()
@click.option('-p','--path',type=str,help = 'path to generated star file')
@click.option('-L','--image-size',type=int,help='image size in pixels')
@click.option('-r','--rank',type=int,help='rank of volume covariance')
@click.option('-n','--num-ims',type=int,help='number of images to generate')
@click.option('-v','--volumes-path',type=str,default=None,help='path to directory containing volumes (mrc files) to generate dataset from')
@click.option('--no-ctf',is_flag = True,help='whether to simulate CTFs')
@click.option('--snr',type=float,default=None,help='SNR of simulated dataset (default INF)')
def generate_dummy_dataset(path,image_size,rank,num_ims = 1000,volumes_path=None,no_ctf = False,snr = None):
    pixel_size = 3 * 128 / image_size
    if(volumes_path is None):
        voxels = LegacyVolume(L=image_size,C=(rank+1),dtype=np.float32,).generate()
    else:
        voxels = [Volume.load(os.path.join(volumes_path,vol)) for vol in sorted(os.listdir(volumes_path)) if '.mrc' in vol]
        voxels = Volume(np.concatenate([v.asnumpy() for v in voxels],axis=0)).downsample(image_size)
    if(not no_ctf):
        filters = [RadialCTFFilter(defocus=d,pixel_size=pixel_size) for d in np.linspace(8e3, 2.5e4, 927)]
    else:
        filters = [ArrayFilter(np.ones((image_size,image_size)))]

    mean_voxel = Volume(np.mean(voxels,axis=0))
    images_states = np.arange(0,voxels.shape[0]).repeat(np.ceil(num_ims / voxels.shape[0]))[:num_ims]
    sim = Simulation(n = num_ims , vols = voxels,unique_filters=filters,offsets=0,amplitudes=1,states=images_states)

    var = np.var((sim.images[:] - sim.vol_forward(mean_voxel,0,sim.n)).asnumpy())

    if(snr is not None):
        noise_var = var / snr
        noise_adder = WhiteNoiseAdder(noise_var)
        sim = Simulation(n = num_ims, vols = voxels,unique_filters=filters,noise_adder=noise_adder,offsets=0,amplitudes=1,states=images_states)


    #The starfile format ASPIRE saves the file as is a little different.
    particles_block = {k[1:] : v for k,v in sim._metadata.items()} #Remove an uncessery underscore prefix since starfile.write adds one anyway
    optics_block = {'rlnOpticsGroup': ['1'], 'rlnOpticsGroupName': ['opticsGroup1'], 
                'rlnAmplitudeContrast' : [str(particles_block['rlnAmplitudeContrast'][0])] , 'rlnSphericalAberration' : [str(particles_block['rlnSphericalAberration'][0])] , 'rlnVoltage' : [str(particles_block['rlnVoltage'][0])],
                'rlnImagePixelSize': [str(pixel_size)],'rlnImageSize': [str(image_size)], 'rlnImageDimensionality': ['2']}

    file_dir = os.path.split(path)[0]
    mrcs_file = os.path.splitext(os.path.split(path)[1])[0] + '.mrcs'
    particles_block['rlnImageName'] = [f'{i+1}@{mrcs_file}' for i in range(num_ims)]
    particles_block['rlnOpticsGroup'] = [1 for i in range(num_ims)]
    
    star_dict = {'optics' : pd.DataFrame(optics_block),'particles' : pd.DataFrame(particles_block)}
    starfile.write(star_dict,os.path.join(path,os.path.splitext(os.path.split(path)[1])[0] + '.star'))

    #Same thing for the mrcs file
    sim.images[:].save(os.path.join(path,mrcs_file),overwrite=True)

    mean_voxel.save(os.path.join(path,'mean_volume.mrc'),overwrite=True)
    voxels.save(os.path.join(path,'ground_truth_states.mrc'),overwrite=True)
    
    





if __name__ == "__main__":
    generate_dummy_dataset()