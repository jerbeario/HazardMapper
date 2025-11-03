#!/bin/bash
#SBATCH -n 1
#SBATCH -t 5-00:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multi

hazard="flood"

# Run the LR
experiment="LR_full"
python HazardMapper/model.py -n $experiment -z $hazard -a LR -e 20 --map

# Run the RF
experiment="RF_full"
python HazardMapper/model.py -n $experiment -z $hazard -a RF -e 20 --map

# Run the MLP
experiment="MLP_full"
python HazardMapper/model.py -n $experiment -z $hazard -a MLP -e 20 --map

# Run the CNN
experiment="CNN_full_sweep"
python HazardMapper/model.py -n $experiment -z $hazard -a CNN -e 20 --sweep
experiment="CNN_full"
python HazardMapper/model.py -n $experiment -z $hazard -a CNN -e 20 --map --explain

# Run the SimpleCNN
experiment="SimpleCNN_full_sweep"
python HazardMapper/model.py -n $experiment -z $hazard -a SimpleCNN -e 20 --sweep
experiment="SimpleCNN_full"
python HazardMapper/model.py -n $experiment -z $hazard -a SimpleCNN -e 20 --map --explain

# Run the SpatialAttentionCNN
experiment="SpatialAttentionCNN_full_sweep"
python HazardMapper/model.py -n $experiment -z $hazard -a SpatialAttentionCNN -e 20 --sweep
experiment="SpatialAttentionCNN_full"
python HazardMapper/model.py -n $experiment -z $hazard -a SpatialAttentionCNN -e 20 --map --explain

# Run the CNN_GAP
experiment="CNN_GAP_full_sweep"
python HazardMapper/model.py -n $experiment -z $hazard -a CNN_GAP -e 20 --sweep
experiment="CNN_GAP_full"
python HazardMapper/model.py -n $experiment -z $hazard -a CNN_GAP -e 20 --map --explain

# Run the CNN
experiment="CNN_full_sweep"
python HazardMapper/model.py -n $experiment -z $hazard -a CNN_GAPatt -e 20 --sweep
experiment="CNN_full"
python HazardMapper/model.py -n $experiment -z $hazard -a CNN_GAPatt -e 20 --map --explain
wait