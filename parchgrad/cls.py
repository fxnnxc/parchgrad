from .base import ParchGradBase
import torch 
import scipy.stats
import torch.nn.functional as F 

class ParchGradCLS(ParchGradBase):
    def __init__(self, model, **kwargs):
        super().__init__(self, model, **kwargs)
        self.mask_function = make_mask_function()
        self.convolutions = None 
        self.mean = None 
        self.shapiro = None 
        self.std = None 
        
    def prepare_parchgrad(self, cls_p_values, device, **kwargs):
        for conv in self.convolutions:
            cls_p_values = cls_p_values.pop(0).to(device)   
            cls_p_values = torch.nan_to_num(cls_p_values, 1)
            conv.cls_p_values = cls_p_values
        
    def make_mask_function(self):
        def mask_function(module, sample_idx, **kwargs):
            # ---- required values 
            cls_p_values = module.c[:,module.cls[sample_idx]] # cls only for the first c
            # ----
            
            if kwargs.get("quantile") is not None:
                feasible_n = max(1, int(len(cls_p_values)*kwargs.get("quantile")))
                alpha = torch.kthvalue(cls_p_values, feasible_n, keepdim=True).values.item()

            h_mask = (cls_p_values <= alpha) *  (module.SHAPIRO < 0.05) # * (gap > mean) # This is unessary if we use alternative H1 greater in p_values
            l_mask = ~h_mask
            return l_mask, h_mask  
        
        
        return mask_function
        
