import numpy as np
import os
from umap import UMAP
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pickle
import click
from external.recovar_utils import recovarReconstructFromEmbedding



def create_scatter_figure(coords,cluster_coords,labels):
    fig = plt.figure()
    plt.scatter(coords[:,0],coords[:,1],s=0.1,c=labels)
    x_min, x_max = np.percentile(coords[:,0], [0.5, 99.5])
    x_delta = x_max - x_min
    y_min, y_max = np.percentile(coords[:,1], [0.5, 99.5])
    y_delta = y_max-y_min
    for i in range(cluster_coords.shape[0]):
        plt.annotate(str(i),(cluster_coords[i,0],cluster_coords[i,1]),fontweight='bold')
    plt.xlim(x_min - 0.1 * x_delta, x_max + 0.1 * x_delta)
    plt.ylim(y_min - 0.1 * y_delta, y_max + 0.1 * y_delta)
    return fig

def create_umap_figure(umap_coords,cluster_coords,labels=None):
    fig = create_scatter_figure(umap_coords,cluster_coords,labels)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    return {'umap' : fig}

def create_pc_figure(pc_coords,cluster_coords,labels=None,num_pcs = 5):
    figures = {}
    for i in range(num_pcs):
        for j in range(i+1,num_pcs):
            fig = create_scatter_figure(pc_coords[:,[i,j]],cluster_coords[:,[i,j]],labels)
            plt.xlabel(f'PC {i}')
            plt.ylabel(f'PC {j}')
            figures[f'pc_{i}_{j}'] = fig

    return figures

@click.command()
@click.option('-i','--result-data',type=str,help='path to pkl output of the algorithm')
@click.option('-o','--output-dir',type=str,help='directory to store analysis output (same directory as result_data by default)',default=None)
@click.option('--analyze-with-gt',is_flag=True,help='whether to also perform analysis with embedding from gt eigenvolumes (if availalbe)')
@click.option('--num-clusters',type=int,default=40,help='number of k-means clusters used to reconstruct from embedding')
@click.option('--skip-reconstruction',is_flag=True,help='whether to skip reconstruction of k-means cluster centers')
@click.option('--gt-labels',default=None,help='path to pkl file containing gt labels. if provided used for coloring embedding figures')
def analyze_cli(result_data,output_dir=None,analyze_with_gt=False,num_clusters=40,skip_reconstruction=False,gt_labels=None):
    analyze(result_data,output_dir,analyze_with_gt=analyze_with_gt,num_clusters=num_clusters,skip_reconstruction=skip_reconstruction,gt_labels=gt_labels)

def analyze(result_data,output_dir=None,analyze_with_gt=False,num_clusters=40,skip_reconstruction=False,gt_labels=None):
    with open(result_data,'rb') as f:
        data = pickle.load(f)

    if(gt_labels is not None):
        with open(gt_labels,'rb') as f:
            gt_labels = pickle.load(f)
    else:
        gt_labels = None

    if(output_dir is None):
        output_dir = os.path.join(os.path.split(result_data)[0],'output')
    os.makedirs(output_dir,exist_ok=True)

    coords_keys = ['coords_est']
    coords_covar_inv_keys = ['coords_covar_inv_est']
    analysis_output_dir = ['analysis']
    figure_prefix = ['']
    if(analyze_with_gt):
        if(data.get('coords_GT') is not None):
            coords_keys.append('coords_GT')
            coords_covar_inv_keys.append('coords_covar_inv_GT')
            analysis_output_dir.append('analysis_gt')
            figure_prefix.append('gt_')
        else:
            print('analyze_with_gt was set to True but coords_GT is not present in result_data - skipping analysis with gt coordinates')

    figure_paths = {}
    for coords_key,coords_covar_inv_key,analysis_dir,fig_prefix in zip(coords_keys,coords_covar_inv_keys,analysis_output_dir,figure_prefix):
        analysis_data,figures = analyze_coordinates(data[coords_key],num_clusters,gt_labels)
        fig_path = save_analysis_result(os.path.join(output_dir,analysis_dir),analysis_data,figures)
        figure_paths.update({fig_prefix+k : v for k,v in fig_path.items()})
        if(not skip_reconstruction):
            #TODO: handle GT reconstruction - right now recovarReconstructomFromEmbedding will still use est embedding
            recovarReconstructFromEmbedding(result_data,os.path.join(output_dir,analysis_dir),analysis_data['cluster_coords'])

    return figure_paths

def analyze_coordinates(coords,num_clusters,gt_labels):

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(coords)
    cluster_coords = kmeans.cluster_centers_

    reducer = UMAP(n_components=2)
    umap_coords = reducer.fit_transform(coords)
    umap_cluster_coords = reducer.transform(cluster_coords)

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

    return data,figures

def save_analysis_result(dir,data,figures):
    os.makedirs(dir,exist_ok=True)

    with open(os.path.join(dir,'data.pkl'),'wb') as f:
        pickle.dump(data,f)

    figure_paths = {}
    for fig_name,fig in figures.items():
        figure_path = os.path.join(dir,f'{fig_name}.jpg')
        fig.savefig(figure_path)
        figure_paths[fig_name] = figure_path

    return figure_paths


if __name__ == "__main__":
    analyze_cli()