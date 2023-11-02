

def get_hook_wrapper(name, model, method):
    from parchgrad.models.resnet import ResNetParchGradCLS, ResNetParchGradINS
    from parchgrad.models.vgg import VGGParchGradCLS, VGGParchGradINS
    from parchgrad.models.efficientnet import EfficientNetParchGradCLS, EfficientNetParchGradINS
    
    if name == "resnet18":
        if method == "ins":
            return ResNetParchGradINS(model)
        elif method in ['cls', 'normal']:
            return ResNetParchGradCLS(model)
    elif name == "vgg16":
        if method == "ins":
            return VGGParchGradINS(model)
        elif method in ['cls', 'normal']:
            return VGGParchGradCLS(model)
    elif name == "efficient_b0":
        if method == "ins":
            return EfficientNetParchGradINS(model)
        elif method in ['cls', 'normal']:
            return EfficientNetParchGradCLS(model)

    else:
        raise ValueError(f"Not implemented: {name} in get_hook_wrapper")