import starfile
import sys
import os
import click
import numpy as np
from aspire.source import RelionSource
from aspire.image import Image
from tqdm import tqdm
'''
This script generates a pre-processed star and mrcs file.Pre processing includes centring and downsampling images.
'''

def replaceMRCSName(img_name,new_filename):
    img_num,_ = img_name.split('@')
    return f'{img_num}@{new_filename}'

def preprocess_statfile(input_star,output_star,mrcs_file,new_imageSize):

    star = starfile.read(input_star)
    star['optics']['rlnImagePixelSize'] = star['optics']['rlnImagePixelSize'] * star['optics']['rlnImageSize']/new_imageSize
    star['optics']['rlnImageSize'] = int(new_imageSize)

    star['particles'] = star['particles'].drop(columns=['rlnOriginXAngst','rlnOriginYAngst','rlnOriginX','rlnOriginY','rlnImageSize','rlnImagePixelSize'],errors='ignore')

    #Replace MRCS file name in star file image reference
    mrcs_filename = os.path.split(mrcs_file)[1]
    star['particles']['rlnImageName'] = [f'{i+1}@{mrcs_filename}' for i in range(len(star['particles']))]
    starfile.write(star,output_star)


def preprocess_mrcs(input_star,output_mrcs,image_size):
    star = starfile.read(input_star)
    pixel_size = star['optics']['rlnImagePixelSize'].loc[0]
    orig_image_size = star['optics']['rlnImageSize'].loc[0]
    source = RelionSource(input_star,pixel_size = pixel_size)

    if('rlnOriginXAngst' in star['particles'].columns):
        shifts = np.array([star['particles'].rlnOriginXAngst,star['particles'].rlnOriginYAngst]).T / pixel_size
    elif('rlnOriginX' in star['particles'].columns):
        shifts = np.array([star['particles'].rlnOriginX,star['particles'].rlnOriginY]).T
    images = source.images[:]
    print('Loaded images')
    preprocessed_images = Image(np.zeros((images.shape[0],image_size,image_size),dtype=images.dtype))
    batch_size = 10240
    for i in tqdm(range(0, images.shape[0], batch_size), desc="Preprocessing images"):
        idx = np.arange(i,min(i+batch_size,images.shape[0]))
        ims = images[idx].shift(-shifts[idx])
        #images = images.shift(-shifts)
        if(image_size != orig_image_size):
            preprocessed_images[idx] = ims.downsample(image_size)
        

    preprocessed_images.save(output_mrcs)

@click.command()
@click.option('-i','--input-star',type=str,help='input star file')
@click.option('-o','--output-star',type=str,help='output star file')
@click.option('-l','--imagesize',type=int,help='image size for downsampling')
def preprocess(input_star,output_star,imagesize):
    if(os.path.isdir(output_star)):
        output_star = os.path.join(output_star,os.path.basename(input_star))
    output_mrcs = output_star.replace('.star','.mrcs')
    preprocess_statfile(input_star,output_star,output_mrcs,imagesize)
    preprocess_mrcs(input_star,output_mrcs,imagesize)

if __name__ == "__main__":
    preprocess()