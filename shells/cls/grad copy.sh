
# 
data_path=/data/ImageNet1k
bbox_path=/data/ILSVRC2012_bbox_val

quantile=1.0
alpha=0.05 # not used because of quantile 
p_value_threshold=0.05
layer_ratio=0.5

for input_attrib in grad
do 
for encoder in resnet18 vgg16 efficient_b0  # convnext_tiny
do
for method in cls 
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
done 