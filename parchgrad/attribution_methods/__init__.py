def get_input_attrib(name):
    from .gradient import gradient 
    from .inputgrad import input_gradient
    from .smoothgrad import smoothgrad
    from .integratedgrad import ig 
    
    if name =="grad":
        return gradient
    elif name == 'inputgrad':
        return input_gradient
    elif name == 'smoothgrad':
        return smoothgrad
    elif name =="ig":
        return ig