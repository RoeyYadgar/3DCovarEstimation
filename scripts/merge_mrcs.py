from aspire.source import RelionSource
import click


@click.command()
@click.option('-i','--input-star',type=str,help='input star file')
@click.option('-o','--output-mrcs',type=str,help='output star file')
def merge_mrcs(input_star,output_mrcs):
    source = RelionSource(input_star)
    images = source.images[:]
    images.save(output_mrcs)


if __name__ == "__main__":
    merge_mrcs()