import torch 
import torch.nn as nn
import numpy as np 

def mask(x, attr, ratios, order):   
    assert x.ndim == 3
    original_size = x.size()
    if not isinstance(attr, torch.Tensor):
        torch.tensor(attr)
    attr = attr.flatten()
    X = [] 
    v, index = torch.sort(attr, descending=True if order == "morf" else False, dim=0)    
    for ratio in ratios:
        copy_x = x.detach().clone()
        copy_x = copy_x.reshape(original_size[0], -1)        
        copy_x[:, index[:int(copy_x.size(1)*ratio)]] = 0.0 
        copy_x = copy_x.reshape(*original_size)
        X.append(copy_x)
    X = torch.stack(X)
    return X

def evaluate_attr_with_logits(input, label, attr, model, device, ratios, order, **kwargs):
    assert input.ndim == 3 
    
    logit = model.forward(input.unsqueeze(0).to(device))
    score_orig = nn.functional.softmax(logit, dim = -1)
    prob_orig = score_orig[0, label].item()
    label_hat = torch.argmax(score_orig)
    prob_conf = score_orig[0, label_hat].item()
    
    masked_input = mask(input, attr, ratios, order)
    assert masked_input.ndim == 4 
    
    masked_input_logit = model(masked_input.to(device))
    score_new = nn.functional.softmax(masked_input_logit, dim = -1)
    prob_new = score_new[:, label].detach().cpu().numpy()
    prob_conf_new = score_new[:, label_hat]
    
    metric_aopc = (prob_orig - prob_new)  #AOPC : drop of probability
    metric_lodds = np.log(prob_new / (prob_orig +1e-10)) #LODDS
    metric_fracdiff = torch.abs(prob_conf - prob_conf_new).detach().cpu().numpy() #difference of logits for predicted label  
    
    y_hat = masked_input_logit.argmax(dim=-1)
    acc = (y_hat == label).detach().cpu().numpy().astype(np.int8)
    return metric_aopc, metric_lodds, acc, metric_fracdiff