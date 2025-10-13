#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate HazardMapper

hazard="flood"
experiment="first_test"

# Run the model 
# Example with SimpleCNN architecture and 10 epochs

python HazardMapper/model.py -n $experiment -z $hazard -a SimpleCNN -e 10 --map 
wait