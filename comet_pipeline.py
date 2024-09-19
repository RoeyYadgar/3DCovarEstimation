import os
import click
import comet_ml
from aspire.storage import StarFile
from workflow import covar_workflow,covar_processing


@click.command()
@click.option('-n','--name',type=str,help = 'name of wandb run')
@click.option('-s','--starfile',type=str, help='path to star file.')
@click.option('-r','--rank',type=int, help='rank of covariance to be estimated.')
@click.option('-w','--whiten',is_flag = True,default=True,help='wether to whiten the images before processing')
@click.option('--noise-estimator',type=str,default = 'anisotropic',help='noise estimator (white/anisotropic) used to whiten the images')
@click.option('--disable-comet',is_flag = True,default = False,help='wether to disable logging of run to comet')
@click.option('--batch-size',type=int,help = 'training batch size')
@click.option('--max-epochs',type=int,help = 'number of epochs to train')
@click.option('--lr',type=float,help= 'training learning rate')
@click.option('--reg',type=float,help='regularization scaling')
@click.option('--gamma-lr',type=float,help = 'learning rate decay rate')
def run_pipeline(name,starfile,rank,whiten,noise_estimator,disable_comet,**training_kwargs):
    if(not disable_comet):
        image_size = int(float(StarFile(starfile)['optics']['_rlnImageSize'][0]))
        run_config  = {'image_size' : image_size, 'rank' : rank,'starfile' : starfile,'whiten' : whiten,'noise_estimator' : noise_estimator}
        exp = comet_ml.Experiment(project_name="3d_cov",parse_args=False)
        exp.set_name(name)
        exp.log_parameters(run_config)
        #TODO : add to parameters image size. pass training arguments to workflow
    training_kwargs = {k : v for k,v in training_kwargs.items() if v is not None}
    data_dict, figure_dict,training_data,training_kwargs = covar_workflow(starfile,rank,whiten=whiten,noise_estimator=noise_estimator,**training_kwargs)

    if(not disable_comet):
        result_dir = os.path.join(os.path.split(starfile)[0],'result_data')
        exp.log_parameters(training_kwargs)
        for fig_name,fig in figure_dict.items():
            exp.log_image(image_data = os.path.join(result_dir,'result_figures',f'{fig_name}.jpg'),name=fig_name)
            
        metrics = {"frobenius_norm_error" : training_data['log_fro_err'][-1],
                   "eigen_vector_cosine_sim" : training_data['log_cosine_sim'][-1],
                   "eigenvals_GD" : data_dict["eigenvals_GD"],"eigenval_est" : data_dict["eigenval_est"]}
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