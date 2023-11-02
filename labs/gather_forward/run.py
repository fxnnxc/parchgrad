
from parchgrad.models.pretrained_models import get_pretrained_model
from parchgrad import get_hook_wrapper
from parchgrad.datasets import get_default_transform
from parchgrad.datasets import IMAGENET_MEAN, IMAGENET_STD, get_datasets
import argparse
import os 
import torch 
import time 
from tqdm import tqdm 
import pickle 


parser = argparse.ArgumentParser()
parser.add_argument("--encoder")
parser.add_argument("--data-path")

args = parser.parse_args()
base_dir = f'outputs/{args.encoder}'
save_dir = os.path.join(base_dir, 'pickles')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- dataset 
encoder = args.encoder
model = get_pretrained_model(encoder)
torch.save(model, f"{base_dir}/model.pt")
model.to("cuda:0")
model.eval()


wrapper = get_hook_wrapper(encoder, model, 'cls')  # just use cls to gather forward hiddens 
wrapper.set_hook_modules(wrapper.all_convolutions)
transform = get_default_transform(wrapper.resize, wrapper.crop, IMAGENET_MEAN, IMAGENET_STD)
_, valid_dataset = get_datasets('imagenet1k', args.data_path, transform)

start_time = time.time()
pbar = tqdm(range(len(valid_dataset)))
tensors_for_relu = [[] for i in range(len(wrapper.hook_modules))]   
tensors = [[] for i in range(len(wrapper.hook_modules))]  
labels = []      
logits = []

for index in pbar:
    x = valid_dataset[index][0].unsqueeze(0).to("cuda:0")
    acts = wrapper.forward(x, modify_gradient=False)
    for i, module in enumerate(wrapper.hook_modules):
        gap = module.gap.clone().detach().cpu()
        relu_ratio = module.relu_ratio.detach().clone().cpu()
        tensors[i].append(gap)
        tensors_for_relu[i].append(relu_ratio)
        
    labels.append(valid_dataset[index][1])
    logits.append(acts.squeeze(0).detach().cpu())
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))         
    pbar.set_description(f"🧪:[{save_dir}] E:({index/len(valid_dataset):.2f}) D:({duration})]")    
    
print("-------------------------------------------------")  
for i, module in enumerate(wrapper.hook_modules):
    tensors[i] = torch.stack(tensors[i])
    tensors_for_relu[i] = torch.stack(tensors_for_relu[i])
    
for t in tensors:
    print(t.size())
with open(os.path.join(save_dir, f"gap.pkl"), 'wb') as f:
    pickle.dump(tensors, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"labels.pkl"), 'wb') as f:
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"logits.pkl"), 'wb') as f:
    pickle.dump(logits, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"relu_ratio.pkl"), 'wb') as f:
    pickle.dump(tensors_for_relu, f, pickle.HIGHEST_PROTOCOL)   
    