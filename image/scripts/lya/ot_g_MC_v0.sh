#!/bin/bash


python run/inference_inverse.py \
    --guide_method MC \
    --problem deblurring \
    --mc_batch_size 256 \
    --start_time 0 \
    --problem deblurring \
    --delta 0.2 \
    --version ES \
    --flow_type ot

    
python run/inference_inverse.py \
    --guide_method MC \
    --problem inpainting \
    --mc_batch_size 256 \
    --start_time 0 \
    --problem inpainting \
    --delta 0.2 \
    --version ES \
    --flow_type ot


python run/inference_inverse.py \
    --guide_method MC \
    --problem superresolution \
    --mc_batch_size 256 \
    --start_time 0 \
    --problem superresolution \
    --delta 0.4 \
    --version ES \
    --flow_type ot
