
# 
data_path=/data/ImageNet1k
bbox_path=/data/ILSVRC2012_bbox_val

input_attrib=grad
alpha=1e-10 # not used because of quantile 
p_value_threshold=0.05
method=cls

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2


# for quantile in 0.05 0.1 
# do 
for layer_ratio in 1.0 0.9 0.5 0.3 
do 
    save_name=$variance_conservation'_'$quantile'_'$layer_ratio
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
            --save-name $save_name
# done  
done 
