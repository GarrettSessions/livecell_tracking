#!/bin/bash

# This script submits Slurm jobs for cell segmentation using cellpose

first_frame="$1"
total_frames="$2"
frames_per_job="$3"

# Path to the Slurm script template
slurm_script="batched_livecell_segmentation_launcher.sh"

# Calculate the number of jobs needed
total_frames_remaining=$((total_frames - first_frame + 1))
num_jobs=$((total_frames_remaining / frames_per_job))
if [ $((total_frames_remaining % frames_per_job)) -ne 0 ]; then
    num_jobs=$((num_jobs + 1))
fi

# Submit job for each frame range
for ((i = 0; i < num_jobs; i++)); do
    start_frame=$((first_frame + i * frames_per_job))
    end_frame=$((start_frame + frames_per_job - 1))
    # Make sure end frame does not exceed total frames
    if [ $end_frame -gt $total_frames ]; then
        end_frame=$total_frames
    fi
    sbatch "$slurm_script" "$start_frame" "$end_frame"
done