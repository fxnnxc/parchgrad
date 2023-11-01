
from parchgrad.cls import ParchGradCLS
from parchgrad.ins import ParchGradINS

class ResNetParchGrad(ParchGradINS):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.convs = [] 
        self.convs.append(self.model.conv1)
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:        
            for basic_block in layer:
                for name in ['conv1', 'conv2', 'conv3']:
                    if hasattr(basic_block, name):
                        self.convs.append(getattr(basic_block, name))
        
                
    def set_hook_modules(self, **kwargs):
        while len(self.hook_modules):
            self.hook_modules.pop()
        for layer in [self.model.layer3, self.model.layer4]:
            for basic_block in layer:
                for name in ['conv1', 'conv2', 'conv3']:
                    if hasattr(basic_block, name):
                        conv = getattr(basic_block, name)
                        self.hook_modules.append(conv)
        
        
class ResNetParchGradINS(ParchGradINS, ResNetParchGrad):
    def __init__(self, model, **kwargs):
        super(ParchGradINS, self).__init__(model, **kwargs)
        super(ResNetParchGrad, self).__init__(model, **kwargs)
        
class ResNetParchGradINS(ParchGradCLS, ResNetParchGrad):
    def __init__(self, model, **kwargs):
        super(ParchGradCLS, self).__init__(model, **kwargs)
        super(ResNetParchGrad, self).__init__(model, **kwargs)