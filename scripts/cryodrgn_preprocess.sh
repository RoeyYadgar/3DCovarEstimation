starfile=$1
L=$2
pixel_size=$3
outdir=$4

source ~/.bashrc
conda activate cryodrgn
cryodrgn parse_pose_star $starfile -o $outdir/poses.pkl -D $L
cryodrgn parse_ctf_star $starfile -o $outdir/ctf.pkl -D $L --Apix $pixel_size
conda deactivate
