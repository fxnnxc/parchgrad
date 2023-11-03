
from parchgrad.metric.evaluate_attr_with_logits import evaluate_attr_with_logits
from parchgrad.metric.context_score import compute_in_out_ratio
import torch 
def evaluate_attribution_all(input, 
                            label, 
                            attr, 
                            model, 
                            device, 
                            bbox, 
                            ratios
                            ):
    
    # ------- evaluate logits -----------
    results = {}
    attr = torch.nan_to_num(attr, nan=0.0)
    attr = attr.abs()
    
    for order in ['morf' ,'lerf']:
        metric_aopc, metric_lodds, acc, metric_fracdiff = evaluate_attr_with_logits(input, label, attr, model, device, ratios, order,)
        
        results.update({order:{
            'aopc'      : {r:metric_aopc[i].item() for i, r in enumerate(ratios)},
            'lodds'     : {r:metric_lodds[i].item() for i, r in enumerate(ratios)} ,
            'acc'       : {r:int(acc[i]) for i, r in enumerate(ratios)} ,
            'fracdiff'  : {r:metric_fracdiff[i].item() for i, r in enumerate(ratios)},
        }})
        
    # ------- evaluate bbox -----------
        
    xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
    bbox_result_dict = compute_in_out_ratio(attr, xmin, xmax, ymin, ymax)
    results.update(bbox_result_dict)
    return results