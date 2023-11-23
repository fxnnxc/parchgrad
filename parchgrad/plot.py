import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import ListedColormap
from tqdm import tqdm 
import cv2 
from PIL import Image


def process_heatmap(R, my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))):
    power = 1.0
    b = 10*((np.abs(R)**power).mean()**(1.0/power))
    my_cmap[:,0:3] *= 0.99
    my_cmap = ListedColormap(my_cmap)
    return (R, {"cmap":my_cmap, "vmin":-b, "vmax":b, "interpolation":'nearest'} )
    

def quantile_plot(wrapper, input_attrib, valid_dataset, valid_dataset_2,  labels, index, axes,  quantiles, device, flags):
    x = valid_dataset[index][0].to(device)
    y = valid_dataset[index][1]
    cls = torch.tensor([y])
    x_img = valid_dataset_2[index][0]

    ax = next(axes)
    ax.imshow((x_img.permute(1,2,0)))
    # ax.set_ylabel(index)
    label = labels[index//50]
    if 'Welsh' in label:
        label = 'Welsh Corgi'
    ax.set_ylabel(label[:10],)
    ax.set_xticks([])
    ax.set_yticks([])
    imgs = [] 
    attrs = [] 
    y = torch.tensor(valid_dataset[index][1]).to(device).unsqueeze(0)
    for i, wrapper in enumerate([wrapper]):
        my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))
        for j, quantile in tqdm(enumerate(quantiles)):
            flags.quantile = quantile
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
            attr = attr.cpu()
            attr, kwargs  = process_heatmap(attr, my_cmap)
            ax = next(axes)
            ax.imshow((x_img.permute(1,2,0)), alpha=0.8) 
            im = ax.imshow(attr.cpu().numpy(), **kwargs, alpha=0.85)
            
            attrs.append(attr)
            gamma_hats = [] 
            betas = []
            num_H = []
            num_L = []
            new_grad_inputs_nonzero = []
                    
            ax.set_title(f"Q:{quantile}")
            ax.set_xticks([])
            ax.set_yticks([])
            
    attr = input_attrib(wrapper, x, y, modify_gradient=False)
    attr = attr.cpu()
    attr, kwargs  = process_heatmap(attr, my_cmap=my_cmap)
    ax = next(axes)
    img1 = ax.imshow((x_img.permute(1,2,0)), alpha=0.7)
    img2 =ax.imshow(attr.cpu().numpy(), **kwargs, alpha=0.8)
    ax.set_title("original")
    ax.set_xticks([])
    ax.set_yticks([])
    return attrs 


def alpha_plot(wrapper, input_attrib, valid_dataset, valid_dataset_2,  labels, index, axes,  alphas, device, flags):
    x = valid_dataset[index][0].to(device)
    y = valid_dataset[index][1]
    cls = torch.tensor([y])
    x_img = valid_dataset_2[index][0]

    ax = next(axes)
    ax.imshow((x_img.permute(1,2,0)))
    # ax.set_ylabel(index)
    label = labels[index//50]
    if 'Welsh' in label:
        label = 'Welsh Corgi'
    ax.set_ylabel(label[:10],)
    ax.set_xticks([])
    ax.set_yticks([])

    y = torch.tensor(valid_dataset[index][1]).to(device).unsqueeze(0)
    for i, wrapper in enumerate([wrapper]):
        my_cmap=plt.cm.seismic(np.arange(plt.cm.seismic.N))
        
        for j, alpha in tqdm(enumerate(alphas)):
            flags.alpha = alpha
            attr = input_attrib(wrapper, x, y, 
                    cls=y, 
                    modify_gradient=False if flags.method == 'normal' else True,
                    quantile=None,
                    alpha=flags.alpha,
                    p_value_threshold=flags.p_value_threshold,
                    variance_conservation=flags.variance_conservation,
                    exact_variance=flags.exact_variance,
                    gamma_infinity=flags.gamma_infinity,
                    enable_forward_hook=True if flags.method == 'ins' else False, 
                    )
            attr = attr.cpu()
            attr, kwargs  = process_heatmap(attr, my_cmap)
            ax = next(axes)
            ax.imshow((x_img.permute(1,2,0)), alpha=0.8)
            ax.imshow(attr.cpu().numpy(), **kwargs, alpha=0.85)
            gamma_hats = [] 
            betas = []
            num_H = []
            num_L = []
            new_grad_inputs_nonzero = []
                    
            ax.set_title(f"A:{alpha}")
            ax.set_xticks([])
            ax.set_yticks([])
            
    attr = input_attrib(wrapper, x, y, modify_gradient=False)
    attr = attr.cpu()
    attr, kwargs  = process_heatmap(attr, my_cmap=my_cmap)
    ax = next(axes)
    ax.imshow((x_img.permute(1,2,0)), alpha=0.7)
    ax.imshow(attr.cpu().numpy(), **kwargs, alpha=0.8)
    ax.set_title("original")
    ax.set_xticks([])
    ax.set_yticks([])
