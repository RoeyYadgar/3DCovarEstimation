import os
import sys
from aspire.volume import Volume


def downsample_volumes(input_dir,output_dir,downsample_size):
    vols = [os.path.join(input_dir,v) for v in os.listdir(input_dir) if '.mrc' in v]
    os.makedirs(output_dir,exist_ok=True)
    for vol in vols:
        v = Volume.load(vol).downsample(downsample_size)
        output_volume_path = os.path.join(output_dir,os.path.split(vol)[1])
        v.save(output_volume_path)



if __name__ == "__main__":
    downsample_volumes(sys.argv[1],sys.argv[2],int(sys.argv[3]))