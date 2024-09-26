import sys
import pickle
import os
import starfile
import numpy as np
import pandas as pd
from aspire.image.image import load_mrc,Image


if __name__ == "__main__":
    starfile_path = sys.argv[1]
    params_path = sys.argv[2]
    pixel_size = sys.argv[3]
    image_size = int(sys.argv[4])




    dataset_dir, _ = os.path.split(starfile_path)
    starfile_data = starfile.read(starfile_path)
    params_data = pd.read_csv(params_path,delim_whitespace=True)
    num_ims = len(starfile_data)

    #starfile_data['rlnAngleRot'] = params_data['PSI']
    #starfile_data['rlnAngleTilt'] = params_data['THETA']
    #starfile_data['rlnAnglePsi'] = params_data['PHI']

    starfile_data['rlnAngleRot'] = params_data['PHI']
    starfile_data['rlnAngleTilt'] = params_data['THETA']
    #starfile_data['rlnAngleRot'] = params_data['THETA']
    #starfile_data['rlnAngleTilt'] = params_data['PHI']
    starfile_data['rlnAnglePsi'] = params_data['PSI']
    starfile_data['rlnOriginXAngst'] = params_data['SHX'] * float(pixel_size) #TODO: validate this is needed
    starfile_data['rlnOriginYAngst'] = params_data['SHY'] * float(pixel_size)
    starfile_data['rlnMicrographName'] = [int(1) for i in range(num_ims)]
    starfile_data['rlnOpticsGroup'] = [int(1) for i in range(num_ims)]
    starfile_data['rlnPhaseShift'] = [0 for i in range(num_ims)]
    
    if(len(sys.argv) >= 6):
        label_pkl_file = sys.argv[5]
        with open(label_pkl_file,'rb') as fid:
            labels = pickle.load(fid)
        starfile_data['rlnClassNumber'] = labels

    ampContrast = starfile_data['rlnAmplitudeContrast'][0]
    sphereAbber = starfile_data['rlnSphericalAberration'][0]
    voltage = starfile_data['rlnVoltage'][0]
    optics_block = {'rlnOpticsGroup': ['1'], 'rlnOpticsGroupName': ['opticsGroup1'],
                    'rlnAmplitudeContrast' : [str(ampContrast)] , 'rlnSphericalAberration' : [str(sphereAbber)] , 'rlnVoltage' : [str(voltage)],
                    'rlnImagePixelSize': [str(pixel_size)],'rlnImageSize' : [str(image_size)], 'rlnImageDimensionality': ['2']}
    star_dict = {'optics' : pd.DataFrame(optics_block),'particles' : starfile_data}
    mrcs_filename = starfile_data['rlnImageName'][0].split('@')[1]
    starfile.write(star_dict,os.path.join(dataset_dir,mrcs_filename.replace('.mrcs','.star')))





    