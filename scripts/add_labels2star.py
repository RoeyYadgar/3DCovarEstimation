import starfile
import pickle
import click

@click.command()
@click.option('-s','--star-file',type=str,help='starfile path')
@click.option('-l','--labels',type=str,help='labels pkl path')
def addlabels(star_file,labels):
    star = starfile.read(star_file)
    star['particles']['rlnClassNumber'] = pickle.load(open(labels,'rb')).astype(int)
    starfile.write(star,star_file)


if __name__ == '__main__':
    addlabels()