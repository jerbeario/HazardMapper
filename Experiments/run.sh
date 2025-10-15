#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate HazardMapper

hazard="flood"
experiment="l=3"

# Run the model 
# Example with SimpleCNN architecture and 10 epochs

python HazardMapper/model.py -n $experiment -z $hazard -a CNN_GAP -e 10 
wait