import pickle
import os
import argparse


def add_args():
    parser = argparse.ArgumentParser(description='Convert RECOVAR model format into own format')
    parser.add_argument('input_dir',help='Recovar result directory')

    return parser

def convert_recovar_model_output(args):
    with open(os.path.join(args.input_dir,'model/params.pkl'),'rb') as f:
        params = pickle.load(f)
    with open(os.path.join(args.input_dir,'model/embeddings.pkl'),'rb') as f:
        embeddings = pickle.load(f)

    recovar_zdims = params['input_args'].zdim
    #TODO: add eigen est
    results = {
        'eigenval_est' : params['s'],
        'coords_est' : embeddings['zs'][max(recovar_zdims)],
        'coords_covar_inv_est' : embeddings['cov_zs'][max(recovar_zdims)],
        'starfile' : params['input_args'].particles.replace('.mrcs','.star')
    }

    with open(os.path.join(args.input_dir,'recorded_data.pkl'),'wb') as f:
        pickle.dump(results,f)


if __name__ == "__main__":
    convert_recovar_model_output(add_args().parse_args())
