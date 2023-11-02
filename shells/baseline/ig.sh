
# 
data_path=/data/ImageNet1k
bbox_path=/data/ILSVRC2012_bbox_val

input_attrib=grad
quantile=1.0
alpha=0.05 # not used because of quantile 
p_value_threshold=0.05
layer_ratio=1.0

for encoder in resnet18  efficient_b0  vgg16 # convnext_tiny
do
for method in normal  # cls ins
do

    python labs/evaluate_attribution/run.py \
            --encoder $encoder \
            --data-path $data_path \
            --bbox-path $bbox_path \
            --method $method \
            --input-attrib $input_attrib \
            --p-value-threshold $p_value_threshold \
            --quantile $quantile \
            --layer-ratio $layer_ratio

done 
done 