from parchgrad.hooks import general_forward_hook, make_backward_hook

class ParchGradBase:
    def __init__(self, model, **kwargs):
        self.model = model 
        self.fw_hooks = []
        self.bw_hooks = []
        self.hook_modules = [] 
        self.mask_function = None 
        
        print("We set all module's inplace=False")
        for m in self.model.modules():
            m.inplace = False
        
    def _remove_hook(self):
        while len(self.bw_hooks):
            self.bw_hooks.pop().remove()
        while len(self.fw_hooks):
            self.fw_hooks.pop().remove()
 
     
    def _register_hook(self, modify_gradient, mask_function, enable_forward_hook=True, **kwargs):
        for conv in self.hook_modules:
            if enable_forward_hook:
                self.fw_hooks.append(
                    conv.register_forward_hook(
                        general_forward_hook
                        )
                )
            self.bw_hooks.append(
                conv.register_full_backward_hook(
                    make_backward_hook(
                        modify_gradient, mask_function, **kwargs
                    )
                )
            )
            
    def set_hook_modules(self, modules, verbose=0, **kwargs):
        while len(self.hook_modules):
            self.hook_modules.pop()
        for m in modules:
            self.hook_modules.append(m)
            if verbose>0:
                print(f"  hooked:{m}")
    
    def prepare_parchgrad(self, **kwargs):
        # setting everything required before running parchgrad
        raise NotImplementedError()
        
    def forward(self, x, modify_gradient, **kwargs):
        self._register_hook(modify_gradient, self.mask_function, **kwargs)
        output = self.model(x)
        self._remove_hook()
        return output
 
    def __call__(self, x, **kawrgs):
        return self.forward(x, **kwargs)
    
    
    
    