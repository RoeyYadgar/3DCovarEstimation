#!/usr/bin/env python3
"""Script to sharpen volumes using relion_postprocess.

This script takes a path to a reconstruction output directory (e.g.,
data/scratch_data/empiar10076/downsample_L128/output_r15_pose_opt/final_output2/analysis)
and runs relion_postprocess on each volume directory to get sharpened volumes.
All sharpened volumes are then moved into a new directory called all_volumes_sharpened.

Usage:
python scripts/sharpen_volumes.py --analysis-dir <path> --angpix <pixel_size> [--mask <mask_file>] [--bfac-val <value>]
"""

import argparse
import os
import re
import subprocess


def get_volume_directories(analysis_dir):
    """Get all volume directories in the analysis directory.

    Args:
        analysis_dir: Path to the analysis directory

    Returns:
        List of volume directory paths (e.g., vol0000, vol0001, etc.)
    """
    volume_dirs = []
    for item in os.listdir(analysis_dir):
        if os.path.isdir(os.path.join(analysis_dir, item)) and item.startswith("vol"):
            volume_dirs.append(item)
    return sorted(volume_dirs)


def parse_postprocess_output(output):
    """Parse relion_postprocess output to extract resolution and b-factor.

    Args:
        output: stdout string from relion_postprocess

    Returns:
        Tuple of (resolution, bfactor) or (None, None) if not found
    """
    resolution = None
    bfactor = None

    for line in output.split("\n"):
        # Look for FINAL RESOLUTION line
        if "FINAL RESOLUTION:" in line:
            try:
                # Extract the resolution value from "== FINAL RESOLUTION: 4.7173"
                match = re.search(r"FINAL RESOLUTION:\s*([\d\.]+)", line)
                if match:
                    resolution = match.group(1) + " Å"
            except Exception:
                pass

        # Look for resolution at FSC=0.5 or 0.143
        if "Resolution at FSC=0.500:" in line or "Resolution at FSC=0.143:" in line:
            try:
                # Extract the resolution value (usually in Angstroms)
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith("A"):
                        resolution = part
                        break
            except Exception:
                pass

        # Look for "apply b-factor of:" line (from auto b-factor calculation)
        if "apply b-factor of:" in line:
            try:
                # Extract b-factor value from "== apply b-factor of: -94.7119"
                match = re.search(r"apply b-factor of:\s*([-+]?\d*\.?\d+)", line)
                if match:
                    bfactor = match.group(1) + " Å²"
            except Exception:
                pass
        # Look for other b-factor mentions
        elif "Applying B-factor of" in line or "Applying auto B-factor of" in line:
            try:
                # Extract b-factor value from "Applying B-factor of X Å^2"
                match = re.search(r"B-factor of ([-+]?\d*\.?\d+)", line)
                if match:
                    bfactor = match.group(1) + " Å²"
            except Exception:
                pass
        elif "B-factor applied" in line:
            try:
                # Extract from "B-factor applied: X Å^2"
                match = re.search(r"B-factor applied: ([-+]?\d*\.?\d+)", line)
                if match:
                    bfactor = match.group(1) + " Å²"
            except Exception:
                pass
        elif "Auto B-factor:" in line:
            try:
                # Extract from "Auto B-factor: X"
                match = re.search(r"Auto B-factor: ([-+]?\d*\.?\d+)", line)
                if match:
                    bfactor = match.group(1) + " Å²"
            except Exception:
                pass

    return resolution, bfactor


def run_relion_postprocess(volume_dir, angpix, mask=None, bfac_val=None):
    """Run relion_postprocess on a volume directory.

    Args:
        volume_dir: Path to the volume directory containing half1_unfil.mrc and half2_unfil.mrc
        angpix: Pixel size in Angstroms
        mask: Optional path to mask file
        bfac_val: B-factor value to use (None for auto, otherwise use this value)

    Returns:
        Tuple of (output_file_path, resolution, bfactor) or (None, None, None) if failed
    """
    half1 = os.path.join(volume_dir, "half1_unfil.mrc")
    half2 = os.path.join(volume_dir, "half2_unfil.mrc")

    # Check if half maps exist
    if not os.path.exists(half1) or not os.path.exists(half2):
        print(f"Warning: Half maps not found in {volume_dir}. Skipping.")
        return None, None, None

    # Create output path in the same directory
    volume_name = os.path.basename(volume_dir)
    output_name = f"{volume_name}_postprocess"
    output_path = os.path.join(volume_dir, output_name)

    # Build command
    cmd = ["relion_postprocess", "--i", half1, "--i2", half2, "--o", output_path, "--angpix", str(angpix)]

    if mask:
        cmd.extend(["--mask", mask])

    if bfac_val is None:
        cmd.append("--auto_bfac")
    else:
        cmd.extend(["--adhoc_bfac", str(bfac_val)])

    # Run the command
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully processed {volume_dir}")

        # Parse the output to extract resolution and b-factor
        resolution, bfactor = parse_postprocess_output(result.stdout)

        # Find the output file
        output_file = output_path + ".mrc"
        if not os.path.exists(output_file):
            print(f"Warning: Output file {output_file} not found after running relion_postprocess")
            return None, resolution, bfactor

        return output_file, resolution, bfactor
    except subprocess.CalledProcessError as e:
        print(f"Error running relion_postprocess on {volume_dir}:")
        print(e.stderr)
        return None, None, None


def move_sharpened_volumes(analysis_dir, volume_dirs):
    """Move all sharpened volumes to a new directory called all_volumes_sharpened.

    Args:
        analysis_dir: Path to the analysis directory
        volume_dirs: List of volume directories that were processed
    """
    all_volumes_dir = os.path.join(analysis_dir, "all_volumes_sharpened")
    os.makedirs(all_volumes_dir, exist_ok=True)

    for volume_dir_name in volume_dirs:
        volume_dir = os.path.join(analysis_dir, volume_dir_name)
        sharpened_vol = os.path.join(volume_dir, f"{volume_dir_name}_postprocess_masked.mrc")

        if os.path.exists(sharpened_vol):
            dest = os.path.join(all_volumes_dir, f"{volume_dir_name}.mrc")
            os.rename(sharpened_vol, dest)
            print(f"Moved {sharpened_vol} to {dest}")
        else:
            print(f"Warning: Sharpened volume not found: {sharpened_vol}")


def main():
    parser = argparse.ArgumentParser(
        description="Sharpen volumes using relion_postprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        dest="analysis_dir",
        help="Path to the analysis directory containing volume directories",
    )
    parser.add_argument("--angpix", type=float, required=True, help="Pixel size in Angstroms")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to mask file")
    parser.add_argument(
        "--bfac-val",
        type=float,
        default=None,
        dest="bfac_val",
        help="B-factor value to use. If not specified, uses automatic B-factor correction",
    )

    args = parser.parse_args()

    # Validate analysis directory
    if not os.path.isdir(args.analysis_dir):
        print(f"Error: {args.analysis_dir} is not a valid directory")
        return

    print(f"Processing volumes in: {args.analysis_dir}")
    print(f"Pixel size: {args.angpix} Angstroms")
    if args.bfac_val is None:
        print("B-factor: Auto (automatic correction)")
    else:
        print(f"B-factor: {args.bfac_val}")

    # Get all volume directories
    volume_dirs = get_volume_directories(args.analysis_dir)
    print(f"Found {len(volume_dirs)} volume directories")

    # Process each volume
    processed_volumes = []
    volume_results = []

    for volume_dir_name in volume_dirs:
        volume_dir = os.path.join(args.analysis_dir, volume_dir_name)
        print(f"\nProcessing {volume_dir_name}...")

        output_file, resolution, bfactor = run_relion_postprocess(
            volume_dir, args.angpix, args.mask, bfac_val=args.bfac_val
        )

        if output_file:
            processed_volumes.append(volume_dir_name)
            volume_results.append(
                {
                    "name": volume_dir_name,
                    "resolution": resolution,
                    "bfactor": bfactor if args.bfac_val is None else args.bfac_val,
                }
            )

            # Report the results for this volume
            if resolution:
                print(f"  Resolution: {resolution}")
            if args.bfac_val is None and bfactor:
                print(f"  Auto B-factor: {bfactor}")

    print(f"\n{'='*60}")
    print(f"Successfully processed {len(processed_volumes)} volumes")
    print(f"{'='*60}")

    # Print summary of results
    if volume_results:
        print("\nPost-processing Results Summary:")
        print("-" * 60)
        for result in volume_results:
            print(f"{result['name']}:")
            if result["resolution"]:
                print(f"  Resolution: {result['resolution']}")
            if result["bfactor"]:
                print(f"  B-factor: {result['bfactor']}")
        print("-" * 60)

    # Move all sharpened volumes to all_volumes_sharpened directory
    print("\nMoving sharpened volumes to all_volumes_sharpened...")
    move_sharpened_volumes(args.analysis_dir, processed_volumes)

    print(f"\nDone! All sharpened volumes are in: {os.path.join(args.analysis_dir, 'all_volumes_sharpened')}")


if __name__ == "__main__":
    main()
