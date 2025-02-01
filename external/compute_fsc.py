"""
Example usage
-------------
$ python compute_fsc.py results/ \
            -o fsc_output/ --gt-dir IgG-1D/vols/128_org/ \
            --mask IgG-1D/init_mask/mask.mrc --num-imgs 1000 --num-vols 100
"""
import os
import sys
import argparse
import logging
import pickle

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOTDIR, "methods", "recovar"))
from recovar import dataset, embedding, output

sys.path.append(os.path.join(ROOTDIR, "fsc"))
from CryoBench.metrics.fsc.utils import volumes, conformations, interface
from CryoBench.metrics.fsc import plot_fsc
from recovar_utils import recovarReconstructFromEmbedding

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
    parser.add_argument("--Bfactor", type=float, default=0, help="0")

    return parser


def main(args: argparse.Namespace) -> None:
    #TODO: get Apix from star file
    """Running the script to get FSCs across conformations produced by RECOVAR."""

    results_dump = os.path.join(args.input_dir, "recorded_data.pkl")
    with open(results_dump,'rb') as f:
        result = pickle.load(f)
    zs = result['coords_est']
    num_imgs = int(args.num_imgs) if zs.shape[0] == 100000 else "ribo"
    nearest_z_array = conformations.get_nearest_z_array(
        zs, args.num_vols, num_imgs
    )

    output.mkdir_safe(args.outdir)
    log_file = os.path.join(args.outdir, "run.log")
    if os.path.exists(log_file) and not args.overwrite:
        logger.info("run.log file exists, skipping...")
    else:
        logger.addHandler(logging.FileHandler(log_file))
        logger.info(args)

        recovarReconstructFromEmbedding(results_dump,args.outdir,nearest_z_array,args.n_bins)
        

    # Align output conformation volumes to ground truth volumes using ChimeraX
    if args.align_vols:
        volumes.align_volumes_multi(args.outdir, args.gt_dir, flip=args.flip_align)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            args.outdir,
            args.gt_dir,
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
                args.gt_dir,
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