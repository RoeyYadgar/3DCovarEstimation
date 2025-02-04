import pickle
import argparse
import os
import jax.numpy as jnp
from CryoBench.metrics.neighborhood_similarity.cal_neighb_hit_werror import *


def add_calc_args() -> argparse.ArgumentParser:
    """Command-line interface used for computing neighborhood similarity"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="dir contains weights, config, z")
    parser.add_argument(
        "-o",
        "--outdir",
        default="output_fsc",
        type=os.path.abspath,
        help="Output directory",
    )
    parser.add_argument("--gt-latent", help="Path to pkl file containing latent coordinates of ground truth states")

    return parser


def generate_neighbor_sim_figure(x,y,error,num_points,decimation_factor,figure_path):


    # Create plot
    plt.errorbar(
        x / num_points * decimation_factor * 100,
        y / x * 100,
        yerr=error / x * 100,
        fmt="o",
        markersize=8,
    )
    plt.plot(
        x / num_points * decimation_factor * 100,
        y / x * 100,
        linestyle="-",
        linewidth=2.5,
    )

    plt.xlabel("Neighborhood Radius [%]", fontsize=20)
    plt.ylabel("% of Matching Neighbors", fontsize=20)
    plt.xlim(0, 10)
    plt.ylim(0, 100)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.set_cmap("tab20")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=1200, bbox_inches="tight")

def main(args: argparse.Namespace) -> None:

    results_dump = os.path.join(args.input_dir, "recorded_data.pkl")
    with open(results_dump,'rb') as f:
        result = pickle.load(f)
    coords_est = jnp.array(result['coords_est'].copy())

    with open(args.gt_latent,'rb') as f:
        coords_gt = jnp.array(pickle.load(f).copy())
    
    num_points = coords_est.shape[0]
    neigh_hit_diff_start_k = []
    decimation_factor = 10
    k_neigh_range = np.arange(num_points//100, num_points//10 + 1, num_points//100) // decimation_factor 

    for start in range(0, 5):
        neigh_hit = calculate_neigh_hits_k(start, coords_gt, coords_est, k_neigh_range,decimation_factor=decimation_factor)
        neigh_hit_diff_start_k.append(neigh_hit)

    mean_neigh_hit_k = np.array(neigh_hit_diff_start_k)

    i = -1
    with open(os.path.join(args.outdir,f"neighbor_sim_output.txt"), "w") as file:
        for k in k_neigh_range:
            i += 1
            file.write(
                f"{k} {mean_neigh_hit_k.mean(0)[i]} {mean_neigh_hit_k.std(0)[i]}\n"
            )

    generate_neighbor_sim_figure(k_neigh_range,mean_neigh_hit_k.mean(0),mean_neigh_hit_k.std(0),
                                 num_points,decimation_factor,os.path.join(args.outdir,'neighbor_sim.pdf'))
    generate_neighbor_sim_figure(k_neigh_range,mean_neigh_hit_k.mean(0),mean_neigh_hit_k.std(0),
                                 num_points,decimation_factor,os.path.join(args.outdir,'neighbor_sim.png'))


if __name__ == "__main__":
    main(add_calc_args().parse_args())