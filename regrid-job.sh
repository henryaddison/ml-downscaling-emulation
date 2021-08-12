#!/bin/bash
#
#
#PBS -l select=1:ncpus=1:mem=500M
#PBS -l walltime=0:0:10

conda activate bpdownscaling

set -euo pipefail

cd $HOME/code/ml-downscaling-emulation

python regrid-gcm.py --data /bp1store/geog-tropical/data/UKCP18/ --derived $WORK
