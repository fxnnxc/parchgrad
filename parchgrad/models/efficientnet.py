
from parchgrad.cls import ParchGradCLS
from parchgrad.ins import ParchGradINS

class EfficientNetParchGrad():
    def __init__(self, model, **kwargs):

        self.resize, self.crop = (256, 224)
        
        self.all_convolutions = []
        for m in model.modules():
            if m.__class__.__name__ ==  'Conv2d':
                self.all_convolutions.append(m)
        
    def get_default_hook_convolutions(self, layer_ratio=None):
        if layer_ratio is not None:
            print(f"[warning] layer_ratio version is not implemented for {self.__class__.__name__}")
        selected_convolutions = [] 
        for num in [6,7]:
            layer = self.model.features[num]
            conv = layer[0].block[1][0]
            selected_convolutions.append(conv)
        selected_convolutions.append(self.model.features[8][0])
        return selected_convolutions
        
            
class EfficientNetParchGradINS(ParchGradINS, EfficientNetParchGrad):
    def __init__(self, model, **kwargs):
        ParchGradINS.__init__(self, model, **kwargs)
        EfficientNetParchGrad.__init__(self, model, **kwargs)
        # super(ParchGradINS, self).__init__(model, **kwargs)
        # super(EfficientNetParchGrad, self).__init__(model, **kwargs)
        
class EfficientNetParchGradCLS(ParchGradCLS, EfficientNetParchGrad):
    def __init__(self, model, **kwargs):
        ParchGradCLS.__init__(self, model, **kwargs)
        EfficientNetParchGrad.__init__(self, model, **kwargs)
        # super(ParchGradCLS, self).__init__(model, **kwargs)
        # super(EfficientNetParchGrad, self).__init__(model, **kwargs)
        
