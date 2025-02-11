import argparse
import numpy as np
import mrcfile
import starfile
from pathlib import Path
import os

def filter_starfile(star_path, index_path, output_dir):
    indices_to_remove = set(np.loadtxt(index_path, dtype=int))
    original_mrcs = Path(star_path).with_suffix('.mrcs')
    output_star = os.path.join(output_dir,os.path.split(star_path)[1])
    output_mrcs = os.path.join(output_dir,os.path.split(original_mrcs)[1])
    
    star_data = starfile.read(star_path)
    df = star_data['particles']
    filtered_df = df.drop(indices_to_remove, errors='ignore').reset_index(drop=True)

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
    parser.add_argument('indexfile', type=str, help='Path to index file (np.loadtxt format)')
    parser.add_argument('output_dir', type=str, help='Path to output directory to save star and mrcs files')
    args = parser.parse_args()
    
    filter_starfile(args.starfile, args.indexfile, args.output_dir)

if __name__ == '__main__':
    main()
