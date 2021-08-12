#!/bin/bash
#
#
#PBS -l select=1:ncpus=1:mem=1G
#PBS -l walltime=0:0:10

source $HOME/.profile

conda activate downscaling

set -euo pipefail

cd $HOME/code/ml-downscaling-emulation

python regrid-gcm.py --data /bp1store/geog-tropical/data/UKCP18/ --derived $WORK
