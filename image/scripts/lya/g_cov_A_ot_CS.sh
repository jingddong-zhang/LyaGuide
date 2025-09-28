#!/bin/bash


python run/inference_inverse.py \
    --guide_method nabla_x1_J_x1 \
    --schedule linear_decay \
    --guide_scale 10.0 \
    --problem deblurring \
    --start_time 0 \
    --time_ratio_eps 1e-2 \
    --no-schedule_ratio \
    --no-clamp_x \
    --delta 1.0 \
    --version CS\
    --flow_type ot


python run/inference_inverse.py \
    --guide_method nabla_x1_J_x1 \
    --schedule const \
    --guide_scale 1.0 \
    --problem inpainting \
    --start_time 0 \
    --time_ratio_eps 1e-2 \
    --no-schedule_ratio \
    --no-clamp_x \
    --delta 0.2 \
    --version CS\
    --flow_type ot


python run/inference_inverse.py \
    --guide_method nabla_x1_J_x1 \
    --schedule const \
    --guide_scale 1.0 \
    --problem superresolution \
    --start_time 0 \
    --time_ratio_eps 1e-2 \
    --no-schedule_ratio \
    --no-clamp_x \
    --delta 2.0	 \
    --version CS\
    --flow_type ot
