export source_dir=.
export encoder=resnext51-2

echo source_dir: $source_dir
echo name: $name 

kiml run submit \
    --experiment parchgrad \
    --image gpt2 \
    --instance-type 1A100-16-MO \
    --num-replica 1 \
    --source-directory $source_dir \
    --name $name \
    --dataset imagenet-1 \
    "bash shells/run.sh" 
    

