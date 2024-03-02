
# 
data_path=/data1/bumjin/ImageNet1k
bbox_path=/data1/bumjin/ILSVRC2012_bbox_val

alpha=0.001 # not used because of quantile 
p_value_threshold=0.05
layer_ratio=0.5
for input_attrib in grad ig 
do 
for encoder in vgg16 resnet18 #  efficient_b0  # convnext_tiny
do
for method in cls 
do 
for guided_backprop in True False 
do
save_name='guided_backprop/'$guided_backprop

    python labs/evaluate_attribution/run.py \
            --encoder $encoder \
            --data-path $data_path \
            --bbox-path $bbox_path \
            --method $method \
            --input-attrib $input_attrib \
            --p-value-threshold $p_value_threshold \
            --alpha $alpha \
            --layer-ratio $layer_ratio \
            --save-name $save_name \
            --guided-backprop $guided_backprop 

done 
done 
done 
done 