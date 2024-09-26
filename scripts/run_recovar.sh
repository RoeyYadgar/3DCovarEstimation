mrcfile=$1
outputdir=$2


source ~/.bashrc
conda activate recovar
python ~/recovar/pipeline.py $mrcfile -o $outputdir --ctf ctf.pkl --poses poses.pkl --mask-option none --correct-contrast --low-memory-option
conda deactivate
