import starfile
import sys
import os
'''
Script to generate a star file for preprocessed data:
    creates a copy of a star file,with updated image and pixel size, and removed offsets.
The actual preprocessing of .mrcs file is being done in the matlab script preprocess.m
'''

def replaceMRCSName(img_name,new_filename):
    img_num,_ = img_name.split('@')
    return f'{img_num}@{new_filename}'

new_imageSize = float(sys.argv[2])

input_star = sys.argv[1] + '.star'
output_star = sys.argv[1] + f'_preprocessed_L{int(new_imageSize)}.star'



star = starfile.read(input_star)
star['optics']['rlnImagePixelSize'] = star['optics']['rlnImagePixelSize'] * star['optics']['rlnImageSize']/new_imageSize
star['optics']['rlnImageSize'] = int(new_imageSize)

star['particles'] = star['particles'].drop(columns=['rlnOriginXAngst','rlnOriginYAngst'])

#Replace MRCS file name in star file image reference
mrcs_file = os.path.split(sys.argv[1])[1] + f'_preprocessed_L{int(new_imageSize)}.mrcs'
star['particles']['rlnImageName']= star['particles']['rlnImageName'].apply(lambda s : replaceMRCSName(s,mrcs_file))
starfile.write(star,output_star)

