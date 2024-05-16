#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 128g
#SBATCH -n 12
#SBATCH -t 07-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=gasessio@email.unc.edu
#SBATCH -o 01_cellpose_segmentation.%j.out 
#SBATCH -e 01_cellpose_segmentation.%j.err

module purge
module load anaconda
conda activate /proj/purvislb/users/Garrett/env/GPU_4i_segment
python /proj/purvislb/users/Garrett/scripts/01_cellpose_segmentation.py