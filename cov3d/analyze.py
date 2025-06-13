import numpy as np
import torch
import os
from umap import UMAP
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pickle
import click
from aspire.volume import Volume
from cov3d.recovar_utils import recovarReconstructFromEmbedding
from cov3d.fsc_utils import covar_fsc
from cov3d import utils
from cov3d.trajectory import compute_density,compute_trajectory,pick_trajectory_pairs,find_closet_idx

def get_embedding_reconstruct_func(method):
    methods = { 'recovar' : recovarReconstructFromEmbedding,
                'relion' : utils.relionReconstructFromEmbedding,
                'reprojection' : utils.reprojectVolumeFromEmbedding,
                'relion_disjoint' : utils.relionReconstructFromEmbeddingDisjointSets,
               }
    
    return methods[method]

def create_scatter_figure(coords,cluster_coords,labels,scatter_size=0.1):
    fig = plt.figure()
    plt.scatter(coords[:,0],coords[:,1],s=scatter_size,c=labels)
    x_min, x_max = np.percentile(coords[:,0], [0.5, 99.5])
    x_delta = x_max - x_min
    y_min, y_max = np.percentile(coords[:,1], [0.5, 99.5])
    y_delta = y_max-y_min
    if(cluster_coords is not None):
        for i in range(cluster_coords.shape[0]):
            plt.annotate(str(i),(cluster_coords[i,0],cluster_coords[i,1]),fontweight='bold')
    plt.xlim(x_min - 0.1 * x_delta, x_max + 0.1 * x_delta)
    plt.ylim(y_min - 0.1 * y_delta, y_max + 0.1 * y_delta)
    return fig

def create_hist_figure(coords,cluster_coords=None,**scater_kwargs):
    fig = sns.jointplot(x=coords[:,0],y=coords[:,1],kind='hex',**scater_kwargs).figure
    if(cluster_coords is not None):
        for i in range(cluster_coords.shape[0]):
            plt.annotate(str(i),(cluster_coords[i,0],cluster_coords[i,1]),fontweight='bold')

    x_min, x_max = np.percentile(coords[:,0], [0.5, 99.5])
    x_delta = x_max - x_min
    y_min, y_max = np.percentile(coords[:,1], [0.5, 99.5])
    y_delta = y_max-y_min
    plt.xlim(x_min - 0.1 * x_delta, x_max + 0.1 * x_delta)
    plt.ylim(y_min - 0.1 * y_delta, y_max + 0.1 * y_delta)
    return fig

def create_umap_figure(umap_coords,cluster_coords=None,labels=None,fig_type='scatter',**scatter_kwargs):
    fig = create_scatter_figure(umap_coords,cluster_coords,labels,**scatter_kwargs) if fig_type=='scatter' else create_hist_figure(umap_coords,cluster_coords,**scatter_kwargs)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    return {'umap' : fig}

def create_pc_figure(pc_coords,cluster_coords=None,labels=None,num_pcs = 5,fig_type='scatter',**scatter_kwargs):
    figures = {}
    num_pcs = min(num_pcs,pc_coords.shape[1])
    for i in range(num_pcs):
        for j in range(i+1,num_pcs):
            clust = cluster_coords[:,[i,j]] if cluster_coords is not None else None        
            fig = create_scatter_figure(pc_coords[:,[i,j]],clust,labels,**scatter_kwargs) if fig_type=='scatter' else create_hist_figure(pc_coords[:,[i,j]],clust,**scatter_kwargs) 
            
            plt.xlabel(f'PC {i}')
            plt.ylabel(f'PC {j}')
            figures[f'pc_{i}_{j}'] = fig

    return figures

def create_covar_fsc_figure(fsc):
    fig,axs = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios': [1, 1]})
    im1 = axs[0].imshow(fsc,vmin=0,vmax=1,aspect='auto')
    fsc_mean = np.mean(fsc)
    axs[0].set_title('Covar FSC - entry mean: {:.3f}'.format(fsc_mean))
    axs[0].set_xlabel('Resolution index')
    axs[0].set_ylabel('Resolution index')
    fig.colorbar(im1, ax=axs[0])

    fsc_diag = np.diag(fsc)
    fsc_diag_mean = np.mean(fsc_diag)
    fsc_cutoff = np.max(np.where(fsc > 0.143))
    axs[1].plot(fsc_diag)
    axs[1].set_title('Covar FSC diagonal - mean: {:.3f}, \n Threshold=0.143 cutoff: {:.3f}'.format(fsc_diag_mean, fsc_cutoff))
    axs[1].set_xlabel('Resolution index')
    axs[1].set_ylabel('FSC')
    return fig

def plot_volume_projections(volumes):
    """
    Plot k x 3 projections (mean along axis) for a k x n x n x n tensor of volumes.
    Positive values shown in red, negative in blue, zero as white.
    Adds black horizontal bars between volume rows.
    """
    k = volumes.shape[0]
    fig = plt.figure(figsize=(12, 4 * k + k - 1))  # Add extra height for black bars

    total_rows = 2 * k - 1  # One row per volume, plus black bars between
    spec = gridspec.GridSpec(total_rows, 3, height_ratios=[0.05 if i % 2 else 1 for i in range(total_rows)],
                              hspace=0.0, wspace=0.0)

    for i in range(k):
        vol = volumes[i]

        # Projections
        proj_x = vol.mean(axis=0)  # yz
        proj_y = vol.mean(axis=1)  # xz
        proj_z = vol.mean(axis=2)  # xy
        projections = [proj_x, proj_y, proj_z]

        row_idx = 2 * i  # Actual plot row (even index)

        for j, proj in enumerate(projections):
            ax = fig.add_subplot(spec[row_idx, j])

            vmax = np.percentile(np.abs(proj), 99)
            ax.imshow(proj, cmap='bwr', vmin=-vmax, vmax=vmax, interpolation='none')
            ax.set_title(f'Vol {i+1}, Proj {["X", "Y", "Z"][j]}', fontsize=10)
            ax.axis('off')

        # Add black bar (a blank Axes filled with black)
        if i < k - 1:
            for j in range(3):
                bar_ax = fig.add_subplot(spec[row_idx + 1, j])
                bar_ax.set_facecolor('black')
                bar_ax.set_xticks([])
                bar_ax.set_yticks([])
                bar_ax.axis('off')

    return fig


@click.command()
@click.option('-i','--result-data',type=str,help='path to pkl output of the algorithm')
@click.option('-o','--output-dir',type=str,help='directory to store analysis output (same directory as result_data by default)',default=None)
@click.option('--analyze-with-gt',is_flag=True,help='whether to also perform analysis with embedding from gt eigenvolumes (if availalbe)')
@click.option('--num-clusters',type=int,default=40,help='number of k-means clusters used to reconstruct from embedding')
@click.option('--latent-coords',type=str,default=None,help='path to pkl containing latent coords to be used as cluster centers instead of k-means')
@click.option('--reconstruct-method',type=str,default='recovar',help='which volume reconstruction method to use')
@click.option('--skip-reconstruction',is_flag=True,help='whether to skip reconstruction of k-means cluster centers')
@click.option('--skip-coor-analysis',is_flag=True,help='whether to skip coordinate analysis (kmeans clustering & umap)')
@click.option('--num-trajectories',type=int,default=0,help='Number of trajectories to compute (default 0)')
@click.option('--gt-labels',default=None,help='path to pkl file containing gt labels. if provided used for coloring embedding figures')
def analyze_cli(result_data,**kwargs):
    analyze(result_data,**kwargs)

def analyze(result_data,output_dir=None,analyze_with_gt=False,num_clusters=40,latent_coords=None,reconstruct_method='recovar',skip_reconstruction=False,skip_coor_analysis=False,num_trajectories=0,gt_labels=None):
    with open(result_data,'rb') as f:
        data = pickle.load(f)

    if(gt_labels is not None):
        if(isinstance(gt_labels,str)):
            with open(gt_labels,'rb') as f:
                gt_labels = pickle.load(f)

    if(latent_coords is not None):
        with open(latent_coords,'rb') as f:
            latent_coords = pickle.load(f)

    if(output_dir is None):
        output_dir = os.path.join(os.path.split(result_data)[0],'output')
        print(f'Writing analysis output to {output_dir}')
    os.makedirs(output_dir,exist_ok=True)

    coords_keys = ['coords_est']
    coords_covar_inv_keys = ['coords_covar_inv_est']
    analysis_output_dir = ['analysis']
    figure_prefix = ['']
    eigenvols_keys = ['eigen_est']
    if(analyze_with_gt):
        if(data.get('coords_GT') is not None):
            coords_keys.append('coords_GT')
            coords_covar_inv_keys.append('coords_covar_inv_GT')
            analysis_output_dir.append('analysis_gt')
            figure_prefix.append('gt_')
            eigenvols_keys.append('eigenvectors_GT')
        else:
            print('analyze_with_gt was set to True but coords_GT is not present in result_data - skipping analysis with gt coordinates')

    figure_paths = {}
    for coords_key,_,analysis_dir,fig_prefix,eigenvols_key in zip(coords_keys,coords_covar_inv_keys,analysis_output_dir,figure_prefix,eigenvols_keys):
        if(not skip_coor_analysis):
            analysis_data,figures,umap_reducer = analyze_coordinates(data[coords_key],num_clusters if latent_coords is None else latent_coords,gt_labels)
            figures['eigenvol_projections'] = plot_volume_projections(data.get(eigenvols_key))
            fig_path = save_analysis_result(os.path.join(output_dir,analysis_dir),analysis_data,figures,eigenvols = data.get(eigenvols_key))
            figure_paths.update({fig_prefix+k : v for k,v in fig_path.items()})
            cluster_coords = analysis_data['cluster_coords']
        else:
            cluster_coords = latent_coords
        if(not skip_reconstruction):
            reconstruct_func = get_embedding_reconstruct_func(reconstruct_method)
            reconstruct_func(result_data,os.path.join(output_dir,analysis_dir),cluster_coords)
        if(num_trajectories > 0):
            print('Computing trajectories')
            coords = data[coords_key]
            start,end = pick_trajectory_pairs(cluster_coords,num_trajectories)
            start_ind = [find_closet_idx(coords,s)[1] for s in start]
            end_ind = [find_closet_idx(coords,s)[1] for s in end]
            coords_density,knn_indices = compute_density(coords)
            trajectories = compute_trajectory(coords,coords_density,start_ind,end_ind,knn_indices=knn_indices)

            for i,traj in enumerate(trajectories):
                #Save trajectory figures
                traj_dir = os.path.join(output_dir,analysis_dir,f'trajectory_{i+1}')
                umap_trajectory = umap_reducer.transform(traj)
                figures = {
                    **create_umap_figure(analysis_data['umap_coords'],umap_trajectory,gt_labels),
                    **create_pc_figure(coords,traj,gt_labels)
                }
                save_analysis_result(traj_dir,figures=figures)
                with open(os.path.join(traj_dir,'trajectory.pkl'),'wb') as f:
                    pickle.dump(traj,f)
                reconstruct_func = get_embedding_reconstruct_func(reconstruct_method)
                reconstruct_func(result_data,traj_dir,traj)

                


    if(analyze_with_gt and data.get('eigenvectors_GT') is not None):
        #Compare covariance FSC between ground truth and estimated eigenvectors
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        eigenvecs_est = data['eigen_est'] * (data['eigenval_est']**0.5).reshape(-1,1,1,1)
        eigenvecs_GT = data['eigenvectors_GT'] * (data['eigenvals_GT']**0.5).reshape(-1,1,1,1)
        fsc_result = covar_fsc(torch.tensor(eigenvecs_est,device=torch_device),torch.tensor(eigenvecs_GT,device=torch_device))

        L = eigenvecs_est.shape[-1]
        fsc_result = fsc_result.cpu().numpy()[:L//2,:L//2] 
        covar_fsc_figure = create_covar_fsc_figure(fsc_result)
        figure_path = os.path.join(output_dir,analysis_output_dir[0],f'covar_GT_fsc.jpg')
        covar_fsc_figure.savefig(figure_path)
        figure_paths['covar_GT_fsc'] = figure_path

    return figure_paths

def analyze_coordinates(coords,num_clusters,gt_labels):

    reducer = UMAP(n_components=2)
    umap_coords = reducer.fit_transform(coords)

    if(isinstance(num_clusters,np.ndarray)): #If num_clusters is already the cluster_coords
        cluster_coords = num_clusters
        umap_cluster_coords = reducer.transform(cluster_coords)
    elif(num_clusters != 0):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(coords)
        cluster_coords = kmeans.cluster_centers_
        umap_cluster_coords = reducer.transform(cluster_coords)
    else:
        cluster_coords = None
        umap_cluster_coords = None

    figures = {
        **create_umap_figure(umap_coords,umap_cluster_coords,gt_labels),
        **create_pc_figure(coords,cluster_coords,gt_labels)
    }

    data = {
        'coords' : coords,
        'cluster_coords' : cluster_coords,
        'umap_coords' : umap_coords,
        'umap_cluster_coords' : umap_cluster_coords
    }

    return data,figures,reducer

def save_analysis_result(dir,data=None,figures=None,eigenvols = None):
    os.makedirs(dir,exist_ok=True)

    if(data is not None):
        with open(os.path.join(dir,'data.pkl'),'wb') as f:
            pickle.dump(data,f)

    if(eigenvols is not None):
        for i,vol in enumerate(eigenvols):
            Volume(vol).save(os.path.join(dir, f'eigenvol_pos{i:03d}.mrc'),overwrite=True)
            Volume(-1 * vol).save(os.path.join(dir, f'eigenvol_neg{i:03d}.mrc'),overwrite=True)

    figure_paths = {}
    if(figures is not None):
        for fig_name,fig in figures.items():
            figure_path = os.path.join(dir,f'{fig_name}.jpg')
            fig.savefig(figure_path)
            figure_paths[fig_name] = figure_path

    return figure_paths


if __name__ == "__main__":
    analyze_cli()