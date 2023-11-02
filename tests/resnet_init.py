from parchgrad.models.pretrained_models import get_pretrained_model
from parchgrad.models.resnet import ResNetParchGradINS, ResNetParchGradCLS


model = get_pretrained_model('resnet18')

wrapper1 = ResNetParchGradINS(model)
wrapper2 = ResNetParchGradCLS(model)

wrapper1.prepare_parchgrad(None, None, None, None)

import torch 
x = torch.randn(1, 3, 224, 224)

output = wrapper1(x)
print(output)