import os
import click
from aspire.volume import Volume

@click.command()
@click.option('-v','--vol-path',type=str,help='path to volume mrc file')
def negate_vol_cli(vol_path):
    if(os.path.isfile(vol_path)):
        negate_vol(vol_path)
    elif(os.path.isdir(vol_path)):
        for file in os.listdir(vol_path):
            if(file.endswith('.mrc')):
                negate_vol(os.path.join(vol_path,file))

def negate_vol(vol_path):
    vol = Volume.load(vol_path)
    dir_name,base_name = os.path.split(vol_path)
    file_name,file_ext = os.path.splitext(base_name)

    vol_path_neg = os.path.join(dir_name,f'{file_name}_neg{file_ext}')
    neg_vol = -1*vol
    neg_vol.save(vol_path_neg,overwrite = True)


if __name__ == "__main__":
    negate_vol_cli()