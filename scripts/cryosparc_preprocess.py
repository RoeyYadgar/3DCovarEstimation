import starfile
import sys
import numpy as np

def add_shift(starfile_path):
    s = starfile.read(starfile_path)

    if('rlnOriginXAngst' in s['particles'].columns or 'rlnOriginX' in s['particles']):
        return s
    
    s['particles']['rlnOriginX'] = np.zeros(len(s['particles']))
    s['particles']['rlnOriginY'] = np.zeros(len(s['particles']))
    print(s['particles'])
    return s



if __name__ == "__main__":
    star = add_shift(sys.argv[1])
    starfile.write(star,sys.argv[2])
