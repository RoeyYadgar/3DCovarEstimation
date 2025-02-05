import numpy as np
import os
from umap import UMAP
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pickle
import click



def create_scatter_figure(coords,cluster_coords,labels):
    fig = plt.figure()
    plt.scatter(coords[:,0],coords[:,1],s=0.1,c=labels)
    for i in range(cluster_coords.shape[0]):
        plt.annotate(str(i),(cluster_coords[i,0],cluster_coords[i,1]),fontweight='bold')
    return fig

def create_umap_figure(umap_coords,cluster_coords,labels=None):
    fig = create_scatter_figure(umap_coords,cluster_coords,labels)
    plt.xlabel('UMAP 1')
    plt.xlabel('UMAP 2')
    return {'umap' : fig}

def create_pc_figure(pc_coords,cluster_coords,labels=None,num_pcs = 5):
    figures = {}
    for i in range(num_pcs):
        for j in range(i+1,num_pcs):
            fig = create_scatter_figure(pc_coords[:,[i,j]],cluster_coords,labels)
            plt.xlabel(f'PC {i}')
            plt.xlabel(f'PC {j}')
            figures[f'pc_{i}_{j}'] = fig

    return figures

@click.command()
@click.option('-i','--result-data',type=str,help='path to pkl output of the algorithm')
@click.option('-o','--output-dir',type=str,help='directory to store analysis output (same directory as result_data by default)',default=None)
@click.option('--analyze-with-gt',is_flag=True,help='whether to also perform analysis with embedding from gt eigenvolumes (if availalbe)')
@click.option('--num-clusters',type=int,default=40,help='number of k-means clusters used to reconstruct from embedding')
@click.option('--skip-reconstruction',is_flag=True,help='whether to skip reconstruction of k-means cluster centers')
@click.option('--gt-labels',default=None,help='path to pkl file containing gt labels. if provided used for coloring embedding figures')
def analyze_cli(result_data,output_dir,analyze_with_gt=False,num_clusters=40,skip_reconstruction=False,gt_labels=None):
    analyze(result_data,output_dir,analyze_with_gt=analyze_with_gt,num_clusters=num_clusters,skip_reconstruction=skip_reconstruction,gt_labels=gt_labels)

def analyze(result_data,output_dir,analyze_with_gt=False,num_clusters=40,skip_reconstruction=False,gt_labels=None):
    with open(result_data,'rb') as f:
        data = pickle.load(f)

    with open(gt_labels,'rb') as f:
        gt_labels = pickle.load(f)

    if(output_dir is None):
        output_dir = os.path.join(os.path.split(result_data)[0],'output')
    os.makedirs(output_dir,exist_ok=True)

    analysis_data,figures = analyze_coordinates(data['coords_est'],data['coords_covar_inv_est'],num_clusters,skip_reconstruction,gt_labels)
    save_analysis_result(os.path.join(output_dir,'analysis'),analysis_data,figures)

    if(analyze_with_gt):
        if(data.get('coords_GT') is not None):
            analysis_data_gt,figures_gt = analyze_coordinates(data['coords_GT'],data['coords_covar_inv_GT'],num_clusters,skip_reconstruction,gt_labels)
            save_analysis_result(os.path.join(output_dir,'analysis_gt'),analysis_data_gt,figures_gt)
        else:
            print('analyze_with_gt was set to True but coords_GT is not present in result_data - skipping analysis with gt coordinates')

def analyze_coordinates(coords,coords_covar_inv,num_clusters,skip_reconstruction,gt_labels):

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

    if(not skip_reconstruction):
        #TODO: Call reconstruction script with recovar env
        pass

    return data,figures

def save_analysis_result(dir,data,figures):
    os.makedirs(dir,exist_ok=True)

    with open(os.path.join(dir,'data.pkl'),'wb') as f:
        pickle.dump(data,f)

    for fig_name,fig in figures.items():
        fig.savefig(os.path.join(dir,f'{fig_name}.jpg'))


if __name__ == "__main__":
    analyze_cli()