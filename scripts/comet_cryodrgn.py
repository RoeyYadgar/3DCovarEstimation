import os
import subprocess
import click
import comet_ml

@click.command()
@click.option('-n','--name',type=str,help = 'name of comet run')
@click.option('--alg',type=str,help = 'Which algorithm to use (recovar,cryodrgn)')
@click.option('-m','--mrc',type=str, help='path to mrc file')
@click.option('-z','--zdim',type=int,help='Latent space dimension')
@click.option('--num-epochs',type=int,help='Number of epochs to train CRYODRGN')
@click.option('-l','--labels',default=None,type=str,help='path to pkl labels file')
@click.option('--mask',default='sphere',type=str,help='mask type for recovar')
@click.option('--disable-comet',is_flag = True,default = False,help='wether to disable logging of run to comet')
def run_pipeline(name,alg,mrc,zdim,num_epochs,labels,mask,disable_comet):
    mrcdir = os.path.split(mrc)[0]
    starfile = mrc.replace('.mrcs','.star')
    poses = os.path.join(mrcdir,'poses.pkl')
    ctf = os.path.join(mrcdir,'ctf.pkl')
    output_path = os.path.join(mrcdir,f'{alg}_results')
    if(alg == 'cryodrgn'):
        command = f'cryodrgn train_vae {mrc} --poses {poses} --ctf {ctf} --zdim {zdim} -n {num_epochs} -o {output_path} --multigpu'
        analyze_command = f'cryodrgn analyze {output_path} {num_epochs-1}'
    if(alg == 'recovar'):
        command = f'python ~/recovar/pipeline.py {mrc} --poses {poses} --ctf {ctf} --zdim {zdim} -o {output_path} --mask {mask} --correct-contrast --low-memory-option'
        analyze_command = f'python ~/recovar/analyze.py --zdim {zdim} {output_path}'
    if(not disable_comet):
        run_config  = {'starfile' : starfile, 'zdim' : zdim, 'command' : command, 'analyze_command' : analyze_command}
        exp = comet_ml.Experiment(project_name="3d_cov",parse_args=False)
        exp.set_name(name)
        exp.log_parameters(run_config)
    

    subprocess.run(f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {alg} && {command}'", shell=True)
    subprocess.run(f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {alg} && {analyze_command}'", shell=True)
    

    analyze_dir = os.path.join(output_path,f'analyze.{num_epochs-1}') if alg == 'cryodrgn' else os.path.join(output_path,f'output/analysis_{zdim}/umap')
    if(labels is not None):
        umap_file = os.path.join(analyze_dir,'umap.pkl') if alg == 'cryodrgn' else os.path.join(analyze_dir,'embedding.pkl')
        umap_image = os.path.join(analyze_dir,'umap_labeled.jpg')
        os.system(f'python scripts/umap_figure.py -u {umap_file} -l {labels} -o {umap_image}')
    else:
        umap_image = os.path.join(analyze_dir,'umap.png')
    
    if(not disable_comet):
        exp.log_image(image_data = umap_image,name='umap_coords_est')
        exp.end()
    
if __name__ == "__main__":
    run_pipeline()