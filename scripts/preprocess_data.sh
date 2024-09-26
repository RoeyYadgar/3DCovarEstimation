filename=$1
L=$2
pixel_size=$3

DIR="$( dirname "${BASH_SOURCE[0]}" )"

python "$DIR/preprocess_star.py" $filename $L

matlab -nodisplay -r "addpath('$DIR');preprocess('$filename',$L,$pixel_size); exit;"
