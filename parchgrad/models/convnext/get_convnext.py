# https://github.com/facebookresearch/ConvNeXt

import torch 
from timm.models import create_model 
from .convnext_modeling import * 


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


# print(model)

def get_convnext(name='convnext_tiny'):
    model = create_model(
        name, 
        pretrained=False, 
        num_classes=1000, 
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        # in_22k=True
        )

    finetune=f'untracked/{name}_1k_224_ema.pth'
    checkpoint = torch.load(finetune, map_location='cpu')

    checkpoint_model = None
    for model_key in 'model|module'.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    load_state_dict(model, checkpoint_model, prefix="")
    return model 


def forward_hook(module, input, output):
    print(output.size())

if __name__ == "__main__":
    model = get_convnext('tiny')
    
    # print(output.keys())
    for stage in model.stages:
        for block in stage:
            print(block)
            block.register_forward_hook(forward_hook)
    x = torch.rand(1,3,224,224)
    output = model(x)
    print(output.size())
    