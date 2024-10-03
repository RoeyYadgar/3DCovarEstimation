import os
import click

@click.command()
@click.option('-s','--star-file',type=str,help='path to star file')
@click.option('-L','--image_sizes',type=str,help='List of image sizes to preprocess the data to. spereated by ,')
@click.option('-l','--labels',default=None,type=str,help='path to pkl labels file. Default is ../../gt_latents.pkl in relation to star file')
def prepare_dataset(star_file,image_sizes,labels):
    image_sizes = image_sizes.split(',')
    image_sizes = [int(L) for L in image_sizes]
    stardir = os.path.split(star_file)[0]
    if(labels is None):
        labels = os.path.join(stardir,'../../gt_latents.pkl')

    os.system(f'python scripts/add_labels2star.py -s {star_file} -l {labels}')

    for image_size in image_sizes:
        downsample_dir = os.path.join(stardir,f'downsample_L{image_size}')
        
        os.mkdir(downsample_dir) 
        os.system(f'python scripts/preprocess_star.py -i {star_file} -o {downsample_dir} -l {image_size}')
        os.system(f'bash scripts/cryodrgn_preprocess.sh {star_file} {image_size} {downsample_dir}')




if __name__ == "__main__":
    prepare_dataset()