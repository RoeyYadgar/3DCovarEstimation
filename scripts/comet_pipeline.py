import os
import click
import sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import auc
import comet_ml
from aspire.storage import StarFile
from cov3d.workflow import covar_workflow,workflow_click_decorator
from external.cryobench_analyze import analysis_click_decorator,cryobench_analyze


def log_analysis_output(exp,result_dir,gt_dir,gt_latent):
    if(gt_latent is not None):
        neighbor_sim_output = np.loadtxt(os.path.join(result_dir,'neighbor_sim_output.txt'))
        for i in range(neighbor_sim_output.shape[0]):
            exp.log_metric(name='latent_matching_neighbors_ratio',value=neighbor_sim_output[i,1]/neighbor_sim_output[i,0],step=neighbor_sim_output[i,0])
            exp.log_metric(name='latent_matching_neighbors_std',value=neighbor_sim_output[i,2],step=neighbor_sim_output[i,0])

        exp.log_image(image_data = os.path.join(result_dir,'neighbor_sim.png'),name='neighbor_sim')

    if(gt_dir is not None):
        #This is the same code from cryobench's plot_fsc - used to compute FSC AUC. #TODO: log computed AUC in plot_fsc itself?
        fsc_output_dir = os.path.join(result_dir,'output')
        fsc_dirs = [
            d
            for d in os.listdir(fsc_output_dir)
            if d.startswith("fsc_") and os.path.isdir(os.path.join(fsc_output_dir, d))
        ]
        for fsc_lbl in fsc_dirs:
            subdir = os.path.join(fsc_output_dir, fsc_lbl)
            fsc_files = glob(os.path.join(subdir, "*.txt"))
        
            fsc_list = list()
            auc_lst = list()
            for i, fsc_file in enumerate(fsc_files):
                fsc = pd.read_csv(fsc_file, sep=" ")
                fsc_list.append(fsc.assign(vol=i))
                auc_lst.append(auc(fsc.pixres, fsc.fsc))
            auc_metrics = {f'{fsc_lbl}_mean_auc' : np.nanmean(auc_lst),
                           f'{fsc_lbl}_std_auc' : np.nanstd(auc_lst),
                           f'{fsc_lbl}_median_auc' : np.nanmedian(auc_lst),}
            exp.log_metrics(auc_metrics)
            exp.log_image(image_data = os.path.join(fsc_output_dir,f'{fsc_lbl}.png'),name=fsc_lbl)
            exp.log_image(image_data = os.path.join(fsc_output_dir,f'{fsc_lbl}_means.png'),name=f'{fsc_lbl}_means')

@click.command()
@click.option('-n','--name',type=str,help = 'name of wandb run')
@click.option('--disable-comet',is_flag = True,default = False,help='wether to disable logging of run to comet')
@click.option('--run-analysis',is_flag=True,help='wether to run analysis script after algorithm execution')
@click.option('--gt-dir',type=str,help="Directory of ground truth volumes")
@click.option('--gt-latent',type=str,help="Path to pkl containing ground truth embedding")
@click.option('--num-vols',type=int,help="Number of GT volumes to use for FSC computation")
@workflow_click_decorator
def run_pipeline(name,starfile,rank,whiten,noise_estimator,mask,
                 run_analysis,gt_dir,gt_latent,num_vols=None,
                 disable_comet = False,**training_kwargs):
    if(not disable_comet):
        image_size = int(float(StarFile(starfile)['optics']['_rlnImageSize'][0]))
        run_config  = {'image_size' : image_size, 'rank' : rank,'starfile' : starfile,'whiten' : whiten,'noise_estimator' : noise_estimator}
        run_config.update(training_kwargs)
        run_config['cli_command'] = ' '.join(sys.argv)
        exp = comet_ml.Experiment(project_name="3d_cov",parse_args=False)
        exp.set_name(name)
        exp.log_parameters(run_config)
    training_kwargs = {k : v for k,v in training_kwargs.items() if v is not None}
    data_dict, figure_dict,training_data,training_kwargs = covar_workflow(starfile,rank,whiten=whiten,noise_estimator=noise_estimator,mask=mask,**training_kwargs)

    if(run_analysis):
        result_dir = os.path.join(os.path.split(starfile)[0],'result_data')
        cryobench_analyze(result_dir,gt_dir=gt_dir,gt_latent=gt_latent,num_vols=num_vols,mask=mask if os.path.isfile(mask) else None)
        log_analysis_output(exp,result_dir,gt_dir,gt_latent)

    if(not disable_comet):
        result_dir = os.path.join(os.path.split(starfile)[0],'result_data')
        exp.log_parameters(training_kwargs)
        for fig_name,fig in figure_dict.items():
            exp.log_image(image_data = os.path.join(result_dir,'result_figures',f'{fig_name}.jpg'),name=fig_name)
            
        metrics = {"frobenius_norm_error" : training_data['log_fro_err'][-1],
                   "eigen_vector_cosine_sim" : training_data['log_cosine_sim'][-1],
                   "eigenvals_GD" : data_dict["eigenvals_GD"],"eigenval_est" : data_dict["eigenval_est"],
                   }
        exp.log_metrics(metrics)

        fro_log = [exp.log_metric(name='fro_norm_err',value=v,step=i) for i,v in enumerate(training_data['log_fro_err'])]
        epoch_ind_log = [exp.log_metric(name='log_epoch_ind',value=v,step=i) for i,v in enumerate(training_data['log_epoch_ind'])]

        
        data_artifact = comet_ml.Artifact("produced_data","data")
        data_artifact.add(os.path.join(result_dir,'recorded_data.pkl'))
        exp.log_artifact(data_artifact)
        training_artifact = comet_ml.Artifact("training_data","data")
        training_artifact.add(os.path.join(result_dir,'training_results.bin'))
        exp.log_artifact(training_artifact)

        exp.end()
    
if __name__ == "__main__":
    run_pipeline()