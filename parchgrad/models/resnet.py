
from parchgrad.cls import ParchGradCLS
from parchgrad.ins import ParchGradINS

class ResNetParchGrad():
    def __init__(self, model, **kwargs):

        self.resize, self.crop = (256, 224)
        
        self.all_convolutions = []
        for m in model.modules():
            if m.__class__.__name__ ==  'Conv2d':
                self.all_convolutions.append(m)
        
            
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