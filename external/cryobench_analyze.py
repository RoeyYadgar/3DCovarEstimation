import click
import os
import subprocess

CONDA_ENV = 'recovar'

def run_with_conda_env(command):
    print(f'Running {command}')
    command = f"conda run -n {CONDA_ENV} " + command
    #TODO : print STDOUT live
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,shell=True)

    for line in process.stdout:
        print(line, end='')

    process.wait()

@click.command()
@click.option('-i','--result_dir',type=str,help="Result dir of algorithm's output")
@click.option('--gt-dir',type=str,help="Directory of ground truth volumes")
@click.option('--gt-latent',type=str,help="Path to pkl containing ground truth embedding")
@click.option('--mask',type=str,help="Mask mrc file used for FSC computation")
@click.option('--num-vols',type=int,help="Number of GT volumes to use for FSC computation")
def cryobench_analyze(result_dir,gt_dir,gt_latent,mask,num_vols = None):

    output_dir = os.path.join(result_dir,'output')

    script_path = os.path.join(os.path.dirname(__file__), 'compute_neighbor_sim.py')
    neighb_sim = f"python {script_path} {result_dir} -o {result_dir} --gt-latent {gt_latent}"

    if(num_vols is None):
        num_vols = len(os.listdir(gt_dir))
        print(f"num-vols was not provided. Using all {num_vols} GT volumes from {gt_dir}")


    script_path = os.path.join(os.path.dirname(__file__), 'compute_fsc.py')
    fsc_no_mask = f"python {script_path} {result_dir} -o {output_dir} --gt-dir {gt_dir} --num-vols {num_vols}"
    fsc_mask = fsc_no_mask + f" --mask {mask}"

    commands_to_run = [
        neighb_sim,
        fsc_no_mask,
        fsc_mask
    ]

    [run_with_conda_env(command) for command in commands_to_run]

if __name__ == "__main__":
    cryobench_analyze()