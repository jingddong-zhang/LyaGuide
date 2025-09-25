#!/bin/bash

python run/inference_inverse.py \
    --guide_method MC \
    --problem deblurring \
    --mc_batch_size 256 \
    --start_time 0 \
    --problem deblurring \
    --delta 0.1 \
    --version ES
    

python run/inference_inverse.py \
    --guide_method MC \
    --problem inpainting \
    --mc_batch_size 256 \
    --start_time 0 \
    --problem inpainting \
    --delta 1.0 \
    --version ES


python run/inference_inverse.py \
    --guide_method MC \
    --problem superresolution \
    --mc_batch_size 256 \
    --start_time 0 \
    --problem superresolution \
    --delta 0.8 \
    --version ES

