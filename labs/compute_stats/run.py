
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
from parchgrad.utils import shapiro, welches

parser = argparse.ArgumentParser()
parser.add_argument("--encoder")

args = parser.parse_args()
base_dir = f'outputs/{args.encoder}'
save_dir = os.path.join(base_dir, 'pickles')

with open(os.path.join(save_dir, f"labels.pkl"), 'rb') as f:
    labels = pickle.load(f)   
    
with open(os.path.join(save_dir, f"gap.pkl"), 'rb') as f:
    gaps = pickle.load(f)       

shapiro_p_values = []
for layer in tqdm(range(len(gaps))):
    num_channels = gaps[layer].shape[1]
    shapiro_p_values.append([])
    for channel in range(num_channels):
        gap = gaps[layer][:,channel]
        p_value = shapiro(gap)
        shapiro_p_values[-1].append(p_value['p_value'])
    shapiro_p_values[-1] = torch.tensor(shapiro_p_values[-1])
with open(os.path.join(save_dir, f"shapiro_p_values.pkl"), 'wb') as f:
    pickle.dump(shapiro_p_values, f, pickle.HIGHEST_PROTOCOL)       

# # save the statistics -------------
means = []
stds = [] 
for h in gaps:
    means.append(torch.mean(h, dim=0))
    stds.append(torch.std(h, dim=0))
import os 
with open(os.path.join(save_dir, f"gap_mean.pkl"), 'wb') as f:
    pickle.dump(means, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"gap_std.pkl"), 'wb') as f:
    pickle.dump(stds, f, pickle.HIGHEST_PROTOCOL)   


print("===== SHAPIRO =====")
for layer in shapiro_p_values:
    print(layer.size())
print("Num layers:", len(shapiro_p_values))




# P-values for Welch's test -------------
classes = list(set(labels))
cls_p_values = [torch.zeros(p.size(1), len(classes)) for p in gaps]
labels = torch.tensor(labels)
# P-values for Class-wise -------------

for label in tqdm(classes):
    cls_labels = labels==label 
    non_cls_labels = ~cls_labels
    for layer in range(len(gaps)):
        for channel in range(gaps[layer].size(1)):
            population_cls = h[cls_labels, channel]
            population_remaining = h[non_cls_labels, channel]
            outputs = welches(population_cls, population_remaining)
            cls_p_values[layer][channel, label] = outputs['p_value']
            

with open(os.path.join(save_dir, f"cls_p_values.pkl"), 'wb') as f:
    pickle.dump(cls_p_values, f, pickle.HIGHEST_PROTOCOL)
    
    
print("===== Welches =====")
for layer in cls_p_values:
    print(layer.size())
print("Num layers:",len(cls_p_values))
    