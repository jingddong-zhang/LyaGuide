
for task in deblurring inpainting superresolution; do

    python run/inference_inverse.py \
        --flow_type cfm \
        --batch_size 24 \
        --guide_method MC \
        --problem $task \
        --mc_batch_size 256 \
        --start_time 0 
done