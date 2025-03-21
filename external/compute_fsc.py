"""
Example usage
-------------
$ python compute_fsc.py results/ \
            -o fsc_output/ --gt-dir IgG-1D/vols/128_org/ --gt-labels IgG-1D/gt_latents.pkl\
            --mask IgG-1D/init_mask/mask.mrc --num-vols 100
"""
import os
import sys
import argparse
import logging
import pickle
import numpy as np
from glob import glob

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOTDIR, "methods", "recovar"))

sys.path.append(os.path.join(ROOTDIR, "fsc"))
from CryoBench.metrics.fsc.utils import volumes, interface
from CryoBench.metrics.fsc.utils.volumes import numfile_sortkey
from CryoBench.metrics.fsc import plot_fsc
from cov3d.recovar_utils import recovarReconstructFromEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="(%(levelname)s) (%(filename)s) (%(asctime)s) %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--n-bins",
        type=float,
        default=50,
        dest="n_bins",
        help="number of bins for reweighting",
    )
    parser.add_argument(
        "--gt-labels",
        type=str,
        dest="gt_labels",
        help="Path to pkl file containing ground truth labels"        
    )
    parser.add_argument(
        "--use-gt-dir-as-label",
        type=bool,
        dest='use_gt_dir_label',
        help="Whether to assume GT vols in gt-dir have the same order as images - in this case they will be used instead of gt-labels"
    )

    return parser


def main(args: argparse.Namespace) -> None:
    #TODO: get Apix from star file
    """Running the script to get FSCs across conformations."""

    results_dump = os.path.join(args.input_dir, "recorded_data.pkl")
    with open(results_dump,'rb') as f:
        result = pickle.load(f)
    zs = result['coords_est']
    gt_vols = sorted(glob(os.path.join(args.gt_dir, "*.mrc")), key=numfile_sortkey)

    if('Spike-MD' in args.gt_dir): #Cryobench's Spike-MD dataset is treated differently -  As each image comes from a unique conformation
        args.use_gt_dir_label = True 

    if(not args.use_gt_dir_label):
        with open(args.gt_labels,'rb') as f:
            gt_labels = pickle.load(f)
        unique_labels = np.unique(gt_labels)
        indices_per_unique_state = [np.where(gt_labels == v)[0] for v in unique_labels]
        random_index_per_state = np.array([np.random.choice(index_set) for index_set in indices_per_unique_state])
    else:
        random_index_per_state = np.arange(len(gt_vols))
    
    z_array = zs[random_index_per_state]
    

    z_array = z_array[:args.num_vols]
    gt_vols = gt_vols[:args.num_vols]


    os.makedirs(args.outdir,exist_ok=True)
    log_file = os.path.join(args.outdir, "run.log")
    if os.path.exists(log_file) and not args.overwrite:
        logger.info("run.log file exists, skipping...")
    else:
        logger.addHandler(logging.FileHandler(log_file))
        logger.info(args)

        reconstruction_complete = False
        max_attempts = np.inf
        num_attempts = 0
        while (not reconstruction_complete and num_attempts < max_attempts):
            try:
                recovarReconstructFromEmbedding(results_dump,args.outdir,z_array,args.n_bins,overwrite_vols=False)
                reconstruction_complete = True
            except Exception as e:
                print(f'Reconstruction failed with error {e}. Trying again')
                num_attempts += 1
                import jax
                import gc
                jax.clear_caches()
                gc.collect()
        #from cov3d.utils import relionReconstructFromEmbedding
        #relionReconstructFromEmbedding(results_dump,args.outdir,z_array)
        

    # Align output conformation volumes to ground truth volumes using ChimeraX
    if args.align_vols:
        volumes.align_volumes_multi(args.outdir, gt_vols, flip=args.flip_align)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            args.outdir,
            gt_vols,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=lambda i: os.path.join(
                f"vol{i:04d}", "locres_filtered"
            ),
            num_vols = args.num_vols,
        )

        if args.align_vols:
            volumes.get_fsc_curves(
                args.outdir,
                gt_vols,
                mask_file=args.mask,
                fast=args.fast,
                overwrite=args.overwrite,
                vol_fl_function=lambda i: os.path.join(
                    f"vol{i:04d}", "locres_filtered"
                ),
                num_vols = args.num_vols,
            )

    
    plot_fsc.main(args)


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())