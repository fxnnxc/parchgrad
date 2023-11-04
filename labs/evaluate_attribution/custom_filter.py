def f1(flag):
    if not flag.success:
        return False 
    elif flag.input_attrib not in ["ig"]:
        return False 
    elif flag.layer_ratio not in [0.5]:
        return False
    elif flag.method not in ["cls"]:
        return False 
    return True 


def f2(flag):
    if not flag.success:
        return False 
    elif flag.input_attrib not in ["grad"]:
        return False 
    elif flag.layer_ratio not in [0.9]:
        return False
    elif flag.method not in ["cls"]:
        return False 
    elif flag.encoder not in ["resnet18"]:
        return False 
    return True 


def no_variance_filter(flag, method, models):
    if not flag.success:
        return False 
    elif flag.input_attrib not in [method]:
        return False 
    elif flag.layer_ratio not in [1.0, 0.9, 0.5, 0.3]:
        return False
    elif flag.quantile not in [0.05, 0.1]:
        return False
    elif flag.method not in ["cls"]:
        return False 
    elif flag.encoder not in models:
        return False 
    return True 
