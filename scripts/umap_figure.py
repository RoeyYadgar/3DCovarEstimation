import pickle
import click
from matplotlib import pyplot as plt

@click.command()
@click.option('-u','--umap',type=str,help = 'Path to pkl umap embedding')
@click.option('-l','--labels',type=str,help= 'Path to pkl labels')
@click.option('-o','--output',type=str,help= 'Figure output path')
def gen_umap_figure(umap,labels,output):
    umap = pickle.load(open(umap,'rb'))
    labels = pickle.load(open(labels,'rb'))

    fig,ax = plt.subplots()
    ax.scatter(umap[:,0],umap[:,1],c = labels, s = 0.1)
    fig.savefig(output)

if __name__ == "__main__":
    gen_umap_figure()