import torch 
import scipy.stats
import torch.nn.functional as F 

def general_forward_hook(module, input, output):
    module.output = output
    if input[0].size(0) == 1:
        output = F.relu(output.clone().detach())
        hw = output.size(-1) * output.size(-2)
        # temp = output.clone().detach()

        gap = F.adaptive_avg_pool2d(output, output_size=1)[0,:,0,0]
        num_relu = (output>0).view(gap.size(0), -1).sum()
        module.relu_ratio = num_relu/hw
        module.relu_position = input[0]>0
        module.gap = gap
    

import math 
def make_backward_hook(modify_gradient, mask_function, gamma_infinity=True, variance_conservation=True, exact_variance=False, **kwargs):
    def backward_conv2d_hook(module, grad_inputs, grad_outputs):
        #modify gradient signals 
        if modify_gradient:
            B = grad_outputs[0].size(0)
            for i in range(B):
                # for each sample 
                l_mask, h_mask = mask_function(module, i, **kwargs)
                module.num_L = l_mask.sum().item() 
                module.num_H = h_mask.sum().item()
                
                if module.num_H  !=0 and module.num_L!=0:
                    if grad_outputs[0][i,:,:,:].size(-1) == 1: # for (1,1) conv
                        M = grad_outputs[0][i,h_mask,:,:].sum(dim=0).abs().item() + 1e-13
                        N = grad_outputs[0][i,l_mask,:,:].sum(dim=0).abs().item() + 1e-13
                    else:
                        M = grad_outputs[0][i,h_mask,:,:].sum(dim=0).var().item() + 1e-13
                        N = grad_outputs[0][i,l_mask,:,:].sum(dim=0).var().item() + 1e-13
                    if gamma_infinity:
                        if variance_conservation:
                            beta = math.sqrt((M+N)/(M))
                        else:
                            beta = 1.0
                        grad_outputs[0][i,h_mask,:,:] *= beta 
                        grad_outputs[0][i,l_mask,:,:] *= 0.0
                    else:
                        gamma_hat = gamma * (N/M)**(1/2) 
                        if exact_variance:
                            std_temp = grad_outputs[0][i].sum(dim=0).var().item()
                            grad_outputs[0][i,h_mask,:,:] *= gamma_hat
                            std_temp2 = grad_outputs[0][i].sum(dim=0).var().item()
                            grad_outputs[0][i,:,:,:] *= std_temp / std_temp2
                        else:
                            if variance_conservation:
                                beta = math.sqrt((M+N)/(M*(gamma_hat**2)+N))
                            else:
                                beta = 1.0
                            grad_outputs[0][i,h_mask,:,:] *= gamma_hat * beta
                            grad_outputs[0][i,l_mask,:,:] *= beta
            
        grad_inputs = [torch.nn.grad.conv2d_input(grad_inputs[0].shape, module.weight, grad_outputs[0],
                        stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)]
        return grad_inputs
    return backward_conv2d_hook