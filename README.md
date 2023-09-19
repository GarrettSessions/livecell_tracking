# livecell_tracking
python pipeline for segmenting, tracking, and quantifying live cell movie

ORIGNAL CREDIT: Kasia Kedziora
https://github.com/fjorka/tracking_pipeline

This pipeline is modified from the original to included new features for very large datasets and a small number of bug fixes.

**This project is for use with livecell microscopy timelapse data saved as TIFFs or ND2 files**

Cell segmentation is performed using CellPose. The provided code is configured to take advantage of CUDA enabled GPUs for faster segmentation. However, CPU based segmentation is possible and may be enabled through
setting the noted flag in notebook 01_cellpose_segmentation. Also note that GPU segmentation is restricted by the amount of video RAM available. Some movies may be too large to use GPU segmentation on low power GPUs.

Tracking is performed using bTrack, and a custom Napari based tracking correction system. Please see the "Training for Napari" movie, which is a recorded training session from Kasia Kedziora,
for details on how to use this system.

**Python Environment Selection**
Use the livecell_GPU_segment environment for the following notebooks: <div style="page-break-after: always;"></div>
00_prepare_tracking <div style="page-break-after: always;"></div>
01_cellpose_segmentation <div style="page-break-after: always;"></div>
02_labels2df_regionprops <div style="page-break-after: always;"></div>

Use the livecell_tracking environment for the following notebooks: <div style="page-break-after: always;"></div>
03_df2tracks_bTrack <div style="page-break-after: always;"></div>
03_df2tracks_bTrack_divide_series **Note** This notebook is unmodified from the original and may no longer be needed. Only use if the downscaling features are insufficient <div style="page-break-after: always;"></div>
04_tracking_correction <div style="page-break-after: always;"></div>
postCorrection_visual
