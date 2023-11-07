
from parchgrad.cls import ParchGradCLS
from parchgrad.ins import ParchGradINS

class VGGParchGrad():
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
            selected_convolutions = []
            num = len(self.model.features)
            start_num = int(num*0.5)
            print("[INFO] hook upper half convolutions")
            for n in range(start_num, num):
                layer=self.model.features[n]
                if layer.__class__.__name__ == "Conv2d":
                    selected_convolutions.append(layer)
        return selected_convolutions 
    
     
            
class VGGParchGradINS(ParchGradINS, VGGParchGrad):
    def __init__(self, model, **kwargs):
        ParchGradINS.__init__(self, model, **kwargs)
        VGGParchGrad.__init__(self, model, **kwargs)
        # super(ParchGradINS, self).__init__(model, **kwargs)
        # super(VGGParchGrad, self).__init__(model, **kwargs)
        
class VGGParchGradCLS(ParchGradCLS, VGGParchGrad):
    def __init__(self, model, **kwargs):
        ParchGradCLS.__init__(self, model, **kwargs)
        VGGParchGrad.__init__(self, model, **kwargs)
        # super(ParchGradCLS, self).__init__(model, **kwargs)
        # super(VGGParchGrad, self).__init__(model, **kwargs)