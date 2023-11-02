
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# for encoder in resnet18 # efficient_b0 convnext_tiny vgg16
# do
    python labs/compute_stats/run.py \
            --encoder $encoder
# done 