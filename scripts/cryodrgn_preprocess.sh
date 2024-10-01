starfile=$1
L=$2
outdir=$3

source ~/.bashrc
conda activate cryodrgn
cryodrgn parse_pose_star $starfile -o $outdir/poses.pkl -D $L
cryodrgn parse_ctf_star $starfile -o $outdir/ctf.pkl -D $L
conda deactivate
