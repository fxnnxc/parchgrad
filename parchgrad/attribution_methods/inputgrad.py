import torch 
from torch.autograd import Variable

def input_gradient(wrapper, x, y, **kwargs):
    x = Variable(x, requires_grad=True).to(x.device)
    x = x.unsqueeze(0)
    x.retain_grad()
    
    wrapper.model.zero_grad()
    output = wrapper.forward(x, **kwargs)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(x.size(0), output.size()[-1]).zero_().to("cuda").type(x.dtype)
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)
    return ((x*x.grad).abs()).detach().cpu().squeeze(0).mean(axis=0)