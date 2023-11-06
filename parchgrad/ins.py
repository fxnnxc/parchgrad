from .base import ParchGradBase
import torch 
import scipy.stats
import os 
import pickle
import torch.nn.functional as F 

class ParchGradINS(ParchGradBase):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.mask_function = self.make_mask_function()
        
    def prepare_parchgrad(self, base_directory,  device, verbose=0, **kwargs):
        mean = pickle.load(open(os.path.join(base_directory, 'pickles/gap_mean.pkl'), 'rb'))
        std  = pickle.load(open(os.path.join(base_directory, 'pickles/gap_std.pkl'), 'rb'))
        shapiro = pickle.load(open(os.path.join(base_directory, 'pickles/shapiro_p_values.pkl'), 'rb')) 
        mean, std, shapiro
        for conv in self.all_convolutions:
            conv.mean = mean.pop(0).to(device)
            conv.std = std.pop(0).to(device)
            conv.shapiro = shapiro.pop(0).to(device)    
            if verbose>0:
                print(f"mean, std, and shapiro are registered to {conv}")
        
    def make_mask_function(self):
        def mask_function(module, sample_idx, **kwargs):
            # ---- required values 
            mean = module.mean   
            std  = module.std   
            act  = module.output
            shapiro = module.shapiro
            act = F.relu(act)
            # ---- 
            
            gap = F.adaptive_avg_pool2d(act, output_size=1)[sample_idx,:,0,0]
            z_score = (gap - mean)/std                
            p_value = torch.tensor(scipy.stats.norm.sf(z_score.cpu().numpy())).to(act.device)     
            if kwargs.get("quantile") is not None:
                feasible_n = max(1, int(len(p_value)*kwargs.get("quantile")))
                alpha = torch.kthvalue(p_value, feasible_n, keepdim=True).values.item()
   
            h_mask = (p_value < alpha) * ( shapiro < kwargs.get("p_value_threshold"))
            l_mask = ~h_mask
            return l_mask, h_mask  
        
        
        return mask_function
        
