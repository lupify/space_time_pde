#!/bin/bash

set -o errexit # so stops continuing after error

## simulation
#DATETIME=$'date +"%F_%T"'
DATETIME=`date +%F-%H-%M-%S`
#python rayleigh_benard.py -h
echo "$DATETIME"
DIR="outputs/simulations/$DATETIME"
echo "$DIR simulation"
mkdir "$DIR"
mpiexec -n 4 python3 rayleigh_benard.py --dir "$DIR"

# python -m dedalus merge_procs snapshots # not needed in newer version
echo "$DIR conversion to npz file"
python convert_to_npz.py -f "$DIR/snapshots/snapshots_s*.h5" -o "outputs/ml_files/rb2d_$DATETIME.npz"

## visualizations
python plot_slices.py "$DIR/snapshots/snapshots_s*.h5" --output="$DIR/frames"
#bash create_video.sh
ffmpeg -framerate 10 -pattern_type glob -i "$DIR/frames/*.png" -c:v libx264 -r 30 -pix_fmt yuv420p "$DIR/out.mp4"
