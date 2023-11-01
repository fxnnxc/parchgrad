import torchvision
from .convnext.get_convnext import get_convnext

def get_pretrained_model(name):
    if name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
    elif name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    elif name == 'resnet34':
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
    elif name == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = resnet101(weights=weights)
    elif name == 'resnet152':
        from torchvision.models import resnet152, ResNet152_Weights
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = resnet152(weights=weights)
    elif name == 'inceptionv3':
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights)
    elif name== "vgg16":
        from torchvision.models import vgg16_bn, VGG16_BN_Weights
        weights = VGG16_BN_Weights.IMAGENET1K_V1
        model = vgg16_bn(weights=weights)
    elif name == "efficient_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    elif 'convnext' in name:
        model = get_convnext(name)
    else:
        raise ValueError(f"{name} is not implemented")  
    return model 

if __name__ == "__main__":
    import torch
    
    # model = get_pretrained_model("vgg16")
    # print(model)
    # for layer in model.features:
    #     if layer.__class__.__name__ == "Conv2d":
    #         print(layer)
    # exit()
    
    models = [
        # 'convnext_tiny',
        # 'resnet18',
        # 'vgg16',
        # 'inceptionv3',
        'efficient_b0'
        
    ]
    for name in models:
        model = get_pretrained_model(name)
        model.eval()
        model.to("cuda:0")
        if 'inception' in name:
            input = torch.rand(1,3,299,299)
        else:
            input = torch.rand(1,3,224,224)
        output = model(input.to("cuda:0"))
        print(model)
      
        print(output.size())