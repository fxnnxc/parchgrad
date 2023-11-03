
from parchgrad.models.pretrained_models import get_pretrained_model
from parchgrad import get_hook_wrapper
from parchgrad.datasets import get_default_transform
from parchgrad.datasets import IMAGENET_MEAN, IMAGENET_STD, get_datasets
from parchgrad.attribution_methods import get_input_attrib
from parchgrad.bbox.bbox_dataset import BBDataset
from parchgrad.utils import is_there_same_flag
from parchgrad.metric.evaluate_attribution_all import evaluate_attribution_all
import argparse
import os 
import torch 
import time 
import json 
from tqdm import tqdm 
import pickle 
import datetime
from distutils.util import strtobool
from omegaconf import OmegaConf
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument("--encoder")
parser.add_argument("--data-path")
parser.add_argument("--bbox-path")
parser.add_argument("--input-attrib")
parser.add_argument("--fixed-samples", default=None, type=int)

parser.add_argument("--method", choices=['ins', 'cls', 'normal'])
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--layer-ratio",  type=float)

parser.add_argument("--alpha", default=None, type=float)
parser.add_argument("--save-name", default="", type=str)
parser.add_argument("--quantile", default=None, type=float)
parser.add_argument("--p-value-threshold", default=0.05, type=float)
parser.add_argument('--variance-conservation', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,)
parser.add_argument('--exact-variance', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,)
parser.add_argument('--gamma-infinity', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,)

args = parser.parse_args()
base_dir = f'outputs/{args.encoder}'
save_dir = os.path.join(base_dir, args.method, args.save_name, datetime.datetime.now().strftime("%m%d_%H%M%S") )

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

flags = OmegaConf.create(vars(args))
flags.success = False 

if is_there_same_flag(base_dir, flags):
    print("[FAIL] There is success state with the exactly same flag")
    print("---------------------------------------------------------")
    exit()
OmegaConf.save(flags, os.path.join(save_dir, 'config.yaml'))
print(flags)

# --- dataset 
encoder = args.encoder
model = get_pretrained_model(encoder)
torch.save(model, f"{base_dir}/model.pt")
model.to(args.device)
model.eval()

wrapper = get_hook_wrapper(encoder, model, args.method)  # just use cls to gather forward hiddens 
wrapper.prepare_parchgrad(base_directory=base_dir, device=args.device)

remove_n = len(wrapper.all_convolutions) -  int(args.layer_ratio * len(wrapper.all_convolutions))
selected_convolutions = wrapper.all_convolutions[remove_n:]
wrapper.set_hook_modules(selected_convolutions)

transform = get_default_transform(wrapper.resize, wrapper.crop, IMAGENET_MEAN, IMAGENET_STD)
_, valid_dataset = get_datasets('imagenet1k', args.data_path, transform)
label_path= os.path.join(args.data_path, "imagenet_label.json")
ds = BBDataset(args.bbox_path, args.data_path, transform, wrapper.resize, wrapper.crop, label_path=label_path)

# ------------------ input attribution logic -----------------------

input_attrib = get_input_attrib(args.input_attrib)
start_time = time.time()

pbar = tqdm(range(len(valid_dataset)))
full_results = {}
for index in pbar:
    x = valid_dataset[index][0].to(args.device)
    y = torch.tensor(valid_dataset[index][1]).to(args.device).unsqueeze(0)
    attr = input_attrib(wrapper, x, y, 
                        cls=y, 
                        modify_gradient=False if flags.method == 'normal' else True,
                        quantile=flags.quantile,
                        alpha=flags.alpha,
                        p_value_threshold=flags.p_value_threshold,
                        variance_conservation=flags.variance_conservation,
                        exact_variance=flags.exact_variance,
                        gamma_infinity=flags.gamma_infinity,
                        enable_forward_hook=True if flags.method == 'ins' else False, 
                        )
    
    img, info = ds[index]
    sample_results = evaluate_attribution_all(
        input=x,
        label=y,
        model=model,
        attr=attr,
        device=args.device,
        bbox=info['bbox'],
        ratios=[0, 0.1, 0.2,0.3,0.4,0.5]
    )
    full_results[index] = sample_results
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))         
    pbar.set_description(f"🧪:[{save_dir}:{args.input_attrib}:{args.method}] E:({index/len(valid_dataset):.2f}) D:({duration})]")    
    if flags.fixed_samples is not None:
        if index > flags.fixed_samples:
            break 

with open(os.path.join(save_dir, 'evaluation.json'), 'w') as f:
    json.dump(full_results, f, indent=4, sort_keys=True)

flags.success = True 
OmegaConf.save(flags, os.path.join(save_dir, 'config.yaml'))
