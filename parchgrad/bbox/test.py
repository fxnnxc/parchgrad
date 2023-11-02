from template import preprocess_image, compute_in_out_ratio
from deeping.manager import DataManager
from data import BBDataset
import torch 
from tqdm import tqdm
import torchvision.transforms as transforms 
import os 

if not os.path.isdir("results"):
    os.mkdir("results")

BBOX_PATH = "/data3/bumjin_data/ILSVRC2012_bbox_val"
IMAGENET_PATH = "/data3/bumjin_data/ILSVRC2012_val/val"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

model_name = "vgg16"
print("[INFO] Lodding Model: {0}...".format(model_name))
mean = torch.FloatTensor(MEAN).reshape(1,-1,1,1).cuda()
std = torch.FloatTensor(STD).reshape(1,-1,1,1).cuda()

print("[INFO] construcing lrp model for {0}".format(model_name))
lrp = construct_lrp(model_name, "cuda", mean, std)
bbox_dataset = BBDataset(BBOX_PATH)

results = {}
for index in tqdm(range(len(bbox_dataset))):
    info = bbox_dataset[index] 
    bbox = bbox_dataset[index]['bbox']
    
    img = bbox_dataset.get_image(IMAGENET_PATH, info['object_name'], info['file_name'])
    (xmin, xmax, ymin, ymax)  = bbox_dataset.compute_bbox_region_after_resize_and_crop(img, bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax'])

    x = preprocess_image(img)

    output = lrp.forward(x)
    r = output['R'].squeeze(0).cpu().detach().sum(0)
    
    if info['object_name'] not in results:
        results[info['object_name']] = [] 

    results[info['object_name']].append(compute_in_out_ratio(r, xmin, xmax, ymin, ymax))        

dm = DataManager()
dm.save_pickle(results, "results/imagenet_lrp_in_out.pkl")    
print("number of objects: {0}".format(len(results)))