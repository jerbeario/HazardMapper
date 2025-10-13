#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -p rome

#SBATCH --mem=45G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate HazardMapper

hazard="flood"

# Preprocess the data
python HazardMapper/preprocess.py
# Make Partition Map
python HazardMapper/partition.py -z $hazard
# Downscale Maps
python HazardMapper/utils.py

wait