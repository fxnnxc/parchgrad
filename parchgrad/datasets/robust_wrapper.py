# permute and the blur dataset 


from torch.utils.data import Dataset 
import torchvision 
import torchvision.transforms as T
from PIL import ImageFilter
import numpy as np 
import torch 
from einops.layers.torch import Rearrange


 
class JigsawWrapper(Dataset):
    def __init__(self, data, transform, flags, re_size=256, crop_size=224):
        self.shuffle_num = flags.jigsaw.shuffle_num
        self.grid_size = flags.jigsaw.grid_size
        self.data = data
        self.cifar_transform = transform 
        self.resize_crop =  T.Compose([
                                T.Resize(re_size),
                                T.CenterCrop(crop_size)
                                ])
        self.crop_size = crop_size
        self.re_size = re_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x,y = self.data[idx]
        # --------- Jigsaw ------------
        x = self.resize_crop(x)
        x = self.jigsaw_puzzle(x)
        if self.cifar_transform is not None:
            x = self.cifar_transform(x)
        return x,y

    def permute_tokens(self, img_tensor, shuffle_times):
        permute_list = [i for i in range(1, img_tensor.size(0))] 
        for s in range(shuffle_times):
            index_a = np.random.randint(len(permute_list))
            index_b = np.random.randint(len(permute_list))
            permute_list[index_a], permute_list[index_b] = permute_list[index_b], permute_list[index_a]
        index = torch.LongTensor([0]+permute_list)
        img_tensor[:,:] = img_tensor[index, :]
        return img_tensor
    
    def jigsaw_puzzle(self, image):
        grid_size=self.grid_size
        w,h = grid_size, grid_size
        p1,p2 = self.crop_size//w, self.crop_size//h

        image_to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2)
        patch_to_image = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=w, h=h, c=3, p1 = p1, p2 = p2)

        processed_image = T.ToTensor()(image).unsqueeze(0)
        processed_image = image_to_patch(processed_image).squeeze(1).squeeze(0)
        processed_image = self.permute_tokens(processed_image, self.shuffle_num).unsqueeze(0)
        processed_image = patch_to_image(processed_image)
        processed_image = processed_image.squeeze(0)
        processed_image = T.ToPILImage()(processed_image)
        return processed_image
            

    
class BlurWrapper(Dataset):
    def __init__(self, data, transform, flags):
        self.magnitude = flags.blur.magnitude
        self.data = data
        self.cifar_transform = transform 
        self.resize_crop =  T.Compose([
                                T.Resize(256),
                                T.CenterCrop(224)
                                ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x,y = self.data[idx]
        # --------- Blur ------------
        x = self.resize_crop(x)
        x = x.filter(ImageFilter.BoxBlur(self.magnitude))
        if self.cifar_transform is not None:
            x = self.cifar_transform(x)
        return x,y




if __name__ == "__main__":
    dataset = RobustCifarBlur("/data3/bumjin/explain_vision_transformers_with_input_attribution/untracked", False, 1, None )
    
    dataset = RobustCifarPermute("/data3/bumjin/explain_vision_transformers_with_input_attribution/untracked", False, 4, None )
    