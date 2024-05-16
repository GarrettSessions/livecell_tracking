#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 128g
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=gasessio@email.unc.edu
#SBATCH -o 01_batched_livecell_segmentation.%j.out 
#SBATCH -e 01_batched_livecell_segmentation.%j.err

# Load necessary modules
module purge
module load anaconda

# Activate the appropriate conda environment
conda activate /proj/purvislb/users/Garrett/env/GPU_4i_segment

# Call Python script with start and end frame arguments
python /proj/purvislb/users/Garrett/scripts/01_parallel_cellpose_livecell_segmentation.py "$1" "$2"
