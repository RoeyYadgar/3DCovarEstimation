import argparse
import numpy as np
import mrcfile
import starfile
from pathlib import Path
import os
import pickle

def filter_starfile(star_path, index_path, output_dir, filter_index_out=False):
    if(index_path.endswith('.pkl')):
        indices_to_remove = set(pickle.load(open(index_path,'rb')))
    else:
        indices_to_remove = set(np.loadtxt(index_path, dtype=int))
    
    star_data = starfile.read(star_path)
    df = star_data['particles']
    if(not filter_index_out):
        indices_to_remove = set(range(len(df))) - indices_to_remove
    filtered_df = df.drop(indices_to_remove, errors='ignore').reset_index(drop=True)

    
    output_star = os.path.join(output_dir,os.path.split(star_path)[1])
    original_mrcs = Path(star_path).with_suffix('.mrcs')
    output_mrcs = os.path.join(output_dir,os.path.split(original_mrcs)[1])

    # Fix rlnImageName indices
    if 'rlnImageName' in filtered_df.columns:
        filtered_df['rlnImageName'] = [
            f"{i+1}@{os.path.split(output_mrcs)[1]}" for i in range(len(filtered_df))
        ]
    
    if not original_mrcs.exists():
        raise FileNotFoundError(f"Expected .mrcs file {original_mrcs} not found.")
    
    with mrcfile.open(original_mrcs, permissive=True) as mrc:
        filtered_stack = np.delete(mrc.data, list(indices_to_remove), axis=0)
    
    with mrcfile.new(output_mrcs, overwrite=True) as mrc_out:
        mrc_out.set_data(filtered_stack.astype(np.float32))
    
    
    star_data['particles'] = filtered_df
    starfile.write(star_data, output_star)

def main():
    parser = argparse.ArgumentParser(description='Filter .star and .mrcs files based on an index file.')
    parser.add_argument('starfile', type=str, help='Path to input .star file')
    parser.add_argument('indexfile', type=str, help='Path to index file (np.loadtxt or pkl format)')
    parser.add_argument('output_dir', type=str, help='Path to output directory to save star and mrcs files')
    parser.add_argument('--filter-index-out', action='store_true', help='Whether to filter out the indices in the index file')
    args = parser.parse_args()
    
    filter_starfile(args.starfile, args.indexfile, args.output_dir,args.filter_index_out)

if __name__ == '__main__':
    main()
