# evaluate the model with filtering out features 
import torch 
from tqdm import tqdm 
import torchvision.transforms as T
from get_convnext import get_convnext

import torchvision 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])

data_path = '/data3/bumjin/bumjin_data/ILSVRC2012_val'
valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)

model = get_convnext('tiny')
model.eval()
model.to("cuda:0")

from torch.utils.data import DataLoader
eq = 0
dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
pbar = tqdm(dataloader)
for (x, y) in pbar:
    x = x.to("cuda:0")
    y = y.to("cuda:0")
    y_hat = model(x).argmax(dim=-1)
    eq += (y_hat == y).sum()
        
current_performance = eq/len(valid_dataset)
print(current_performance)
