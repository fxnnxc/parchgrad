import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os 
import xml.etree.ElementTree as ET
from PIL import Image
import copy 
from tqdm import tqdm 
import json 

class BBDataset(Dataset):
    def __init__(self, bbox_path, imagenet_path, transforms, resize, crop_size, label_path=None):
        self.path = bbox_path 
        imagenet_path = os.path.join(imagenet_path, "val")
        self.imagenet_path = imagenet_path
        self.data = sorted(os.listdir(os.path.join(self.path)), key=lambda x:int(x.split("_")[2].split(".")[0]))
        self.transforms = transforms
        self.resize = resize
        self.crop_size = crop_size
        self.labels = None 
        
        if label_path is not None:
            self.labels = json.load(open(label_path, "r"))
            # self.label_dict = {self.labels[i]:i for i in range(len(self.labels))}
        
        self.ids_to_labels = {}
        self.id_and_filename_to_index = {}
        
        for i, id in enumerate(sorted(os.listdir(imagenet_path), key=lambda x:int(x[1:]))):
            self.ids_to_labels[id] = i
            for j, file_name in enumerate(sorted(os.listdir(os.path.join(imagenet_path, id)), key=lambda x:int(x.split("_")[2].split(".")[0]))):
                self.id_and_filename_to_index[id + "_" + file_name.split(".")[0]] = i*50 + j
            
         
            
        
    def read_file(self, file_name):
        with open(file_name) as f :
            root = ET.parse(file_name).getroot()
        return root


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = os.path.join(self.path, self.data[idx])
        info = self.read_file(file_name)
        bbox = info.find("object").find("bndbox")

        info_dic = {
            "file_name" : info.find("filename").text,
            "object_name" : info.find("object").find("name").text,
            "label" : self.ids_to_labels[info.find("object").find("name").text],
            "size": {
                "width" : int(info.find("size").find("width").text),
                "height" : int(info.find("size").find("height").text),
                "depth" : int(info.find("size").find("depth").text)
                },
            "bbox":{
                "xmin" : int(bbox.find("xmin").text),
                "xmax" : int(bbox.find("xmax").text),
                "ymin" : int(bbox.find("ymin").text),
                "ymax" : int(bbox.find("ymax").text),
            }
        }
        info_dic['index'] = self.id_and_filename_to_index[info_dic['object_name']+"_"+info_dic['file_name']]
        if self.labels is not None:
            info_dic['label_human'] = self.labels[info_dic['label']]
        
        img = self.get_image(self.imagenet_path, info_dic['object_name'], info_dic['file_name'])
        if self.transforms is not None:
            xmin, xmax, ymin, ymax = (info_dic['bbox']['xmin'],
                                      info_dic['bbox']['xmax'],
                                      info_dic['bbox']['ymin'],
                                      info_dic['bbox']['ymax'],
            )                          
            xmin, xmax, ymin, ymax = self.compute_bbox_region_after_resize_and_crop(img, xmin, xmax, ymin, ymax, self.resize, self.crop_size)
            info_dic['bbox']['xmin'] = xmin 
            info_dic['bbox']['xmax'] = xmax 
            info_dic['bbox']['ymin'] = ymin
            info_dic['bbox']['ymax'] = ymax 
            info_dic['bbox']['ratio'] = ((xmax - xmin) * (ymax - ymin))/(self.crop_size*self.crop_size)
            
            img = self.transforms(img)
        
        return img, info_dic
        
    def get_image(self, path, object_id, file):
        image_path = os.path.join(path, object_id, file + ".JPEG")
        im = Image.open(image_path).convert('RGB')
        return im

    def compute_bbox_region_after_resize_and_crop(self, img, xmin, xmax, ymin, ymax, resize, crop):
        x_1, y_1 = transforms.Resize(resize)(img).size
        x_ratio = img.size[0] / x_1 
        y_ratio = img.size[1] / y_1
        x_crop = (x_1 - crop)//2
        y_crop = (y_1 - crop)//2

        xmin = max(0, int(xmin / x_ratio - x_crop))
        xmax = min(crop, int(xmax / x_ratio - x_crop))
        ymin = max(0, int(ymin / y_ratio - y_crop))
        ymax = min(crop, int(ymax / y_ratio - y_crop))

        return (xmin, xmax, ymin, ymax)

    def collapse_dataset_with_bbox_ratio(self, r_min, r_max, indices=None):
        
        print(f"Collapsing the dataset for {r_min} - {r_max}")
        if indices is None:
            selected = []
            self.data_clone = [] 
            pbar = tqdm(range(len(self.data)))
            for i in pbar:
                img, info = self.__getitem__(i)
                if r_min < info['bbox']['ratio'] < r_max:            
                    selected.append(i)
                    self.data_clone.append(self.data[i])
                pbar.set_description(f"{len(self.data_clone)/(i+1):.3f} data is collected")
            self.data_backup = self.data 
            self.data = self.data_clone
        else:
            self.data_backup = self.data
            self.data = copy.deepcopy(self.data[indices])
        print(f"Collapsed from {len(self.data_backup)} to {len(self.data)}")

