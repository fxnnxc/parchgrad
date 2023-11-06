import torch 
import numpy as np 
import scipy.stats as stats
import os 

def k_th_quantile(tensor, quantile):
    # return the quantile value of a tensor
    assert 0<=quantile<=1
    k_th = int(len(tensor)*quantile)
    if k_th ==0:
        return torch.min(tensor)
    value = torch.kthvalue(tensor, k_th, keepdim=True).values.item()
    return value
    

def welches(target_pop, remaining_pos):
    # check whether the `target_pop` has larger mean than `remaining_pos`
    welches_result = stats.ttest_ind(target_pop, remaining_pos , alternative='greater')
    output = {
        'statistic': welches_result[0],
        'p_value': welches_result[1],
    }
    if np.isnan(output['p_value']):
        output['p_value'] = 1
    return output 


def shapiro(pop):
    # check whether pop is normally distributed
    shapiro_result = stats.shapiro(pop)
    output = {
        'statistic': shapiro_result[0],
        'p_value': shapiro_result[1],
    }
    if np.isnan(output['p_value']):
        output['p_value'] = 1
    return output


from tqdm import tqdm 
from omegaconf import OmegaConf

def is_there_same_flag(base_dir, flag):
    is_same_flag=False 
    for root, dirs, files in tqdm(os.walk(base_dir)):
        is_same_flag = True 
        if 'config.yaml' in files:
            loaded_flag = OmegaConf.load(os.path.join(root, 'config.yaml'))
            if not loaded_flag.success: 
                continue
            for k, v in loaded_flag.items():
                if k =="success":
                    continue 
                if k != "fixed_samples" and getattr(flag, k) != v:
                    is_same_flag = False 
            if is_same_flag:
                return True 
                    
    return False 

