from .base import ParchGradBase
import torch 
import scipy.stats
import pickle
import os 
import torch.nn.functional as F 

class ParchGradCLS(ParchGradBase):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.mask_function = self.make_mask_function()
        self.mean = None 
        self.shapiro = None 
        self.std = None 
        
    def prepare_parchgrad(self, base_directory,  device, verbose=0, **kwargs):
        cls_p_values = pickle.load(open(os.path.join(base_directory, 'pickles/cls_p_values.pkl'), 'rb'))
        shapiro = pickle.load(open(os.path.join(base_directory, 'pickles/shapiro_p_values.pkl'), 'rb')) 
        
        for conv in self.all_convolutions:
            cls_p_value = cls_p_values.pop(0).to(device)   
            conv.cls_p_values = cls_p_value
            conv.shapiro = shapiro.pop(0).to(device)    
            if verbose>0:
                print(f"cls_p_value is registered to {conv}")
        
    def make_mask_function(self):
        def mask_function(module, sample_idx, **kwargs):
            # ---- required values 
            cls_p_values = module.cls_p_values[:,kwargs.get("cls")[0]] # cls only for the first c
            # ----
            
            if kwargs.get("quantile") is not None:
                feasible_n = max(1, int(len(cls_p_values)*kwargs.get("quantile")))
                alpha = torch.kthvalue(cls_p_values, feasible_n, keepdim=True).values.item()
            else:
                alpha = kwargs.get("alpha")
            h_mask = (cls_p_values <= alpha) *  (module.shapiro < kwargs.get("p_value_threshold")) # * (gap > mean) # This is unessary if we use alternative H1 greater in p_values
            if h_mask.sum()==0:
                h_mask = h_mask.fill_(True)
            l_mask = ~h_mask
            return l_mask, h_mask  
        
        
        return mask_function
        
