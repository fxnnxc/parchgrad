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


def f3(flag):
    if not flag.success:
        return False 
    elif flag.input_attrib not in ["grad", 'ig']:
        return False 
    elif flag.layer_ratio not in [1.0]:
        return False
    elif flag.method not in ["cls"]:
        return False 
    elif flag.encoder not in ["resnet18",]:
        return False 
    return True 
