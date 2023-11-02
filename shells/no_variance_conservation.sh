
# 
data_path=/data/ImageNet1k
bbox_path=/data/ILSVRC2012_bbox_val

input_attrib=grad
quantile=0.1
alpha=1e-10 # not used because of quantile 
p_value_threshold=0.05
layer_ratio=0.5
method=cls

for encoder in resnet18 vgg16 efficient_b0 # convnext_tiny
do
for variance_conservation in True False 
do
    python labs/evaluate_attribution/run.py \
            --encoder $encoder \
            --data-path $data_path \
            --bbox-path $bbox_path \
            --method $method \
            --input-attrib $input_attrib \
            --p-value-threshold $p_value_threshold \
            --quantile $quantile \
            --layer-ratio $layer_ratio \
            --variance-conservation $variance_conservation
done 
done 
done 