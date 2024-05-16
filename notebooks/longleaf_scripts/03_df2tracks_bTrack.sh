#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 64g
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=gasessio@email.unc.edu
#SBATCH -o 03_df2tracks_bTrack.%j.out 
#SBATCH -e 03_df2tracks_bTrack.%j.err

module purge
module load anaconda
conda activate /proj/purvislb/users/Garrett/env/livecell_tracking
python /proj/purvislb/users/Garrett/scripts/03_df2tracks_bTrack.py