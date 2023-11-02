data_path=/data/ImageNet1k
bbox_path=/data/ILSVRC2012_bbox_val
method=ins

modify_gradient=False 
input_attrib=grad
quantile=0.1
alpha=0.05 # not used because of quantile 
p_value_threshold=0.05
layer_ratio=0.5

for encoder in resnet18 # efficient_b0 convnext_tiny vgg16
do
for method in  cls ins normal
do

    python labs/evaluate_attribution/run.py \
            --encoder $encoder \
            --data-path $data_path \
            --bbox-path $bbox_path \
            --method $method \
            --modify-gradient $modify_gradient \
            --input-attrib $input_attrib \
            --p-value-threshold $p_value_threshold \
            --quantile $quantile \
            --layer-ratio $layer_ratio

done 
done 