import pickle
import argparse
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

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
    parser.add_argument("--gt-labels", help="Path to pkl file containing ground truth labels")

    return parser



def main(args: argparse.Namespace) -> None:

    results_dump = os.path.join(args.input_dir, "recorded_data.pkl")
    with open(results_dump,'rb') as f:
        result = pickle.load(f)
    coords_est = result['coords_est']

    with open(args.gt_labels,'rb') as f:
        gt_labels = pickle.load(f)

    num_unique_states = len(np.unique(gt_labels))
    kmeans = KMeans(n_clusters=num_unique_states)
    predicted_labels = kmeans.fit_predict(coords_est)

    ami = adjusted_mutual_info_score(gt_labels, predicted_labels)
    print(f"Adjusted Mutual Information: {ami:.4f}")

    # Compute Adjusted Rand Index
    ari = adjusted_rand_score(gt_labels, predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    np.savetxt(os.path.join(args.outdir,'clustering_metrics.txt'),np.array([ami,ari]))
    


if __name__ == "__main__":
    main(add_calc_args().parse_args())