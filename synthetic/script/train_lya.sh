
for pair in "uniform 8gaussian" "8gaussian moon" "circle s_curve"; do
    set -- $pair
    x0_dist=$1
    x1_dist=$2
    python guided_flow/train/lya_s2.py --x0_dist $x0_dist --x1_dist $x1_dist --n_support 128
done


# for pair in "8gaussian moon" "circle s_curve"; do
#     set -- $pair
#     x0_dist=$1
#     x1_dist=$2
#     python guided_flow/train/lya_s2.py --x0_dist $x0_dist --x1_dist $x1_dist
# done