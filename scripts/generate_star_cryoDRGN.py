import sys
import pickle
import os
import numpy as np
import pandas as pd
import starfile
import mrcfile
from aspire.image.image import load_mrc,Image
from aspire.utils.rotation import Rotation
from collections import OrderedDict

def pickle_load(filepath):
    with open(filepath,'rb') as fid:
        return pickle.load(fid)
    

def count_mrcs_images(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        return mrc.data.shape[0]

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    mrcs_files = sys.argv[2].split(',')
    mrcs_files_len = [count_mrcs_images(os.path.join(dataset_dir,f)) for f in mrcs_files]
    #dataset_dir, mrcs_filename = os.path.split(mrcs_path)

    ctf_columns = ["rlnDefocusU",
                    "rlnDefocusV",
                    "rlnDefocusAngle",
                    "rlnVoltage",
                    "rlnSphericalAberration",
                    "rlnAmplitudeContrast",
                    "rlnPhaseShift"]
    ctf = pickle_load(os.path.join(dataset_dir,'ctf.pkl')) #CTF column order : _rlnImageSize,_rlnImagePixelSize,_rlnDefocusU,_rlnDefocusV,_rlnDefocusAngle,_rlnVoltage,_rlnSphericalAberration,   _rlnAmplitudeContrast,    _rlnPhaseShift
    
    poses = pickle_load(os.path.join(dataset_dir,'poses.pkl'))

    num_images = sum(mrcs_files_len)
    #parametsr taken from the first row of ctf parameters matrix, these params are assumed to be constant throughout the whole data
    pixel_size = ctf[0,1]
    image_size = int(ctf[0,0])
    ampContrast = ctf[0,7]
    sphereAbber = ctf[0,6]
    voltage = ctf[0,5]


    optics_block = {'rlnOpticsGroup': ['1'], 'rlnOpticsGroupName': ['opticsGroup1'], 
                    'rlnAmplitudeContrast' : [str(ampContrast)] , 'rlnSphericalAberration' : [str(sphereAbber)] , 'rlnVoltage' : [str(voltage)],
                    'rlnImagePixelSize': [str(pixel_size)],'rlnImageSize': [str(image_size)], 'rlnImageDimensionality': ['2']}
    particles_block = {}
    image_names = []
    for file,file_len in zip(mrcs_files,mrcs_files_len):
        image_names += [f'{str(i+1).zfill(6)}@{file}' for i in range(file_len)]
    particles_block['rlnImageName'] = image_names
    for i,col in enumerate(ctf_columns):
        particles_block[col] = ctf[:,i+2] #first two columns are image and pixel size

    rotvec = Rotation.from_matrix(np.transpose(poses[0],axes=(0,2,1))).angles #Convert rotation matrix into euler vectors
    rotvec_columns = ["rlnAngleRot","rlnAngleTilt","rlnAnglePsi"]
    for i,col in enumerate(rotvec_columns):
        particles_block[col] = rotvec[:,i] / np.pi * 180 #Star file contains rotations in degrees 

    offsets = poses[1]
    offset_columns = ["rlnOriginXAngst","rlnOriginYAngst"]
    for i,col in enumerate(offset_columns):
        particles_block[col] = offsets[:,i] * pixel_size * image_size #offsets is in fraction,required multplication by the pixel and image size to get the offset in angs 
    
    labels_path = os.path.join(dataset_dir,'labels.pkl')
    if(os.path.isfile(labels_path)):
        labels = pickle_load(labels_path)
        particles_block['rlnClassNumber'] = labels.astype(int)

    particles_block['rlnOpticsGroup'] = [int(1) for i in range(num_images)] #TODO: check what OpticsGroup is and why its needed
    particles_block['rlnMicrographName'] = [int(1) for i in range(num_images)]
    particles_block['rlnGroupNumber'] = [int(1) for i in range(num_images)]

    star_dict = {'optics' : pd.DataFrame(optics_block),'particles' : pd.DataFrame(particles_block)}
    starfile.write(star_dict,os.path.join(dataset_dir,mrcs_files[0].replace('.mrcs','.star')))

    
    #Fix MRCS header by reading it with load_mrc and saving it again
    #TODO: only do this if its actually needed
    #Image(load_mrc(mrcs_path)).save(mrcs_path,overwrite = True)