import pickle
import os
import argparse
from aspire.volume import Volume
import numpy as np


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
        'eigenval_est' : params['s'][:max(recovar_zdims)],
        'coords_est' : embeddings['zs'][max(recovar_zdims)],
        'coords_covar_inv_est' : embeddings['cov_zs'][max(recovar_zdims)],
        'starfile' : params['input_args'].particles.replace('.mrcs','.star')
    }

    
    num_eigens = max(recovar_zdims)
    recovar_eigenvectors = Volume(np.zeros((num_eigens,*params['volume_shape'])))
    for i in range(num_eigens):
        volume_path = f'output/volumes/eigen_pos{i:04d}.mrc'
        recovar_eigenvectors[i] = Volume.load(os.path.join(args.input_dir,volume_path)) * np.sqrt(results['eigenval_est'][i])

    results['eigen_est'] = recovar_eigenvectors.asnumpy()[:max(recovar_zdims)]


    try:
        with open(os.path.join(args.input_dir,'../result_data/recorded_data.pkl'),'rb') as f:
            gt_result = pickle.load(f)
        results['eigenvals_GT'] = gt_result['eigenvals_GT']
        results['eigenvectors_GT'] = gt_result['eigenvectors_GT']
        results['coords_GT'] = gt_result['coords_GT']
        results['coords_covar_inv_GT'] = gt_result['coords_covar_inv_GT']
    except:
        pass

    with open(os.path.join(args.input_dir,'recorded_data.pkl'),'wb') as f:
        pickle.dump(results,f)


if __name__ == "__main__":
    convert_recovar_model_output(add_args().parse_args())
