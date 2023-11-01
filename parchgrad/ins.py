from .base import ParchGradBase
import torch 
import scipy.stats
import torch.nn.functional as F 

class ParchGradINS(ParchGradBase):
    def __init__(self, model, **kwargs):
        super().__init__(self, model, **kwargs)
        self.mask_function = make_mask_function()
        self.convolutions = None 
        
    def prepare_parchgrad(self, mean, std, shapiro, device, **kwargs):
        for conv in self.convolutions:
            conv.mean = mean.pop(0).to(device)
            conv.std = std.pop(0).to(device)
            conv.shapiro = torch.tensor(shapiro.pop(0)).to(device)    
        
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
                feasible_n = max(1, int(len(cls_p_values)*kwargs.get("quantile")))
                alpha = torch.kthvalue(cls_p_values, feasible_n, keepdim=True).values.item()
   
            h_mask = (p_value < alpha) *  ( shapiro < 0.05)
            l_mask = ~h_mask
            return l_mask, h_mask  
        
        
        return mask_function
        
