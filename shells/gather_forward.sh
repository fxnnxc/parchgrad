data_path=/data/ImageNet1k

# for encoder in  efficient_b0  resnet18  vgg16 #  #
# do
    python labs/gather_forward/run.py \
            --encoder $encoder \
            --data-path $data_path
# done 