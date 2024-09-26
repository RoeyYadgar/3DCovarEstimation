outputdir=$1
zdim=$2


source ~/.bashrc
conda activate recovar
python ~/recovar/analyze.py $outputdir --zdim=$2
conda deactivate
