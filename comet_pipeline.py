import click
import plotly.tools as tls
import comet_ml
from workflow import covar_workflow,covar_processing
import os

@click.command()
@click.option('-n','--name',type=str,help = 'name of wandb run')
@click.option('-s','--starfile',type=str, help='path to star file.')
@click.option('-r','--rank',type=int, help='rank of covariance to be estimated.')
@click.option('-w','--whiten',is_flag = True,default=True,help='wether to whiten the images before processing')
@click.option('--noise-estimator',type=str,default = 'anisotropic',help='noise estimator (white/anisotropic) used to whiten the images')
@click.option('--disable-comet',is_flag = True,default = False,help='wether to disable logging of run to comet')
def run_pipeline(name,starfile,rank,whiten,noise_estimator,disable_comet):
    if(not disable_comet):
        run_config  = {'rank' : rank,'starfile' : starfile,'whiten' : whiten,'noise_estimator' : noise_estimator}
        exp = comet_ml.Experiment(project_name="3d_cov",parse_args=False)
        exp.set_name(name)
        exp.log_parameters(run_config)
        #TODO : add to parameters image size. pass training arguments to workflow
                
    data_dict, figure_dict,training_data = covar_workflow(starfile,rank,whiten=whiten,noise_estimator=noise_estimator)

    if(not disable_comet):
        for fig_name,fig in figure_dict.items():
            exp.log_figure(figure = fig,figure_name=fig_name)
            
        metrics = {"frobenius_norm_error" : training_data['log_fro_err'][-1],
                   "eigen_vector_cosine_sim" : training_data['log_cosine_sim'][-1],
                   "eigenvals_GD" : data_dict["eigenvals_GD"],"eigenval_est" : data_dict["eigenval_est"]}
        exp.log_metrics(metrics)
        result_dir = os.path.join(os.path.split(starfile)[0],'result_data')
        data_artifact = comet_ml.Artifact("produced_data","data")
        data_artifact.add(os.path.join(result_dir,'recorded_data.pkl'))
        exp.log_artifact(data_artifact)
        training_artifact = comet_ml.Artifact("training_data","data")
        training_artifact.add(os.path.join(result_dir,'training_results.bin'))
        exp.log_artifact(training_artifact)
        exp.end()
    
if __name__ == "__main__":
    run_pipeline()