#!/bin/bash

python run/inference_inverse.py \
    --flow_type cfm \
    --guide_method PiGDM \
    --batch_size 12 \
    --rt_schedule linear_decay \
    --rt_scale 1.0 \
    --schedule const \
    --guide_scale 1.0 \
    --problem deblurring \
    --start_time 0 \
    --time_ratio_eps 1e-2 \
    --no-schedule_ratio \
    --no-clamp_x


python run/inference_inverse.py \
    --flow_type cfm \
    --guide_method PiGDM \
    --batch_size 12 \
    --rt_schedule linear_decay \
    --rt_scale 10.0 \
    --schedule const \
    --guide_scale 10.0 \
    --problem inpainting \
    --start_time 0 \
    --time_ratio_eps 1e-2 \
    --no-schedule_ratio \
    --no-clamp_x


python run/inference_inverse.py \
    --flow_type cfm \
    --guide_method PiGDM \
    --batch_size 12 \
    --rt_schedule linear_decay \
    --rt_scale 0.1 \
    --schedule const \
    --guide_scale 1.0 \
    --problem superresolution \
    --start_time 0 \
    --time_ratio_eps 1e-2 \
    --no-schedule_ratio \
    --no-clamp_x
