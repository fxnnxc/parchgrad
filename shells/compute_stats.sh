
for encoder in resnet18 # efficient_b0 convnext_tiny vgg16
do
    python labs/compute_stats/run.py \
            --encoder $encoder
done 