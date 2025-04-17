import pickle
import numpy as np
import sys
import os
from cryosparc_compute import dataset


#eval $(/scratch/roaiyadgar/cryosparc/cryosparc_master/bin/cryosparcm env)
#export PYTHONPATH='/scratch/roaiyadgar/cryosparc/cryosparc_master'
def coords_cryosparc2np(cs_path):
    d = dataset.Dataset.load(cs_path)

    comp_key = lambda i : f'components_mode_{i}/value'

    coords = d[comp_key(0)]
    i = 1
    while(d.get(comp_key(i)) is not None):
        coords = np.vstack((coords,d[comp_key(i)]))
        i+=1

    coords = coords.T
    

    output_path = os.path.join(os.path.split(cs_path)[0],'coords.pkl')
    print(f'Saving coords {coords.shape} to {output_path}')
    with open(output_path,'wb') as f:
        pickle.dump(coords,f)
    


if __name__ == "__main__":
    coords_cryosparc2np(sys.argv[1])