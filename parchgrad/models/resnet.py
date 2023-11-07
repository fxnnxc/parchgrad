
from parchgrad.cls import ParchGradCLS
from parchgrad.ins import ParchGradINS

class ResNetParchGrad():
    def __init__(self, model, **kwargs):

        self.resize, self.crop = (256, 224)
        
        self.all_convolutions = []
        for m in model.modules():
            if m.__class__.__name__ ==  'Conv2d':
                self.all_convolutions.append(m)
        

    def get_default_hook_convolutions(self, layer_ratio=None):
        if layer_ratio is not None:
            remove_n = len(self.all_convolutions) -  int(layer_ratio * len(self.all_convolutions))
            selected_convolutions = self.all_convolutions[remove_n:]
        else:
            print("[INFO] hook layer3 and layer4 convolutions")
            selected_convolutions = []
            for layer in [self.model.layer3, self.model.layer4]:
                for basic_block in layer:
                    for name in ['conv1', 'conv2', 'conv3']:
                        if hasattr(basic_block, name):
                            conv = getattr(basic_block, name)
                            selected_convolutions.append(conv)
        return selected_convolutions 
    
class ResNetParchGradINS(ParchGradINS, ResNetParchGrad):
    def __init__(self, model, **kwargs):
        ParchGradINS.__init__(self, model, **kwargs)
        ResNetParchGrad.__init__(self, model, **kwargs)
        # super(ParchGradINS, self).__init__(model, **kwargs)
        # super(ResNetParchGrad, self).__init__(model, **kwargs)
        
class ResNetParchGradCLS(ParchGradCLS, ResNetParchGrad):
    def __init__(self, model, **kwargs):
        ParchGradCLS.__init__(self, model, **kwargs)
        ResNetParchGrad.__init__(self, model, **kwargs)
        # super(ParchGradCLS, self).__init__(model, **kwargs)
        # super(ResNetParchGrad, self).__init__(model, **kwargs)