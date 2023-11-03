
# 
data_path=/data/ImageNet1k
bbox_path=/data/ILSVRC2012_bbox_val

input_attrib=grad
quantile=0.1
alpha=1e-10 # not used because of quantile 
p_value_threshold=0.05
method=cls

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

for encoder in resnet18 vgg16 efficient_b0 # convnext_tiny
do
for layer_ratio in 1.0 # 0.5 0.9 
do 
for variance_conservation in False 
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
            --variance-conservation $variance_conservation \
            --fixed-samples 1000
done 
done 
done 
done 