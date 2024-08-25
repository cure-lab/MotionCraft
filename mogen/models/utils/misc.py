def set_requires_grad(nets, requires_grad=False, mode='all'):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    if mode=='all':
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    elif mode=='root':
        # only support to unfreeze root and trans encoder and decoder
        for net in nets:
            if net is not None:
                for name, param in net.named_parameters():
                    if 'trans_embed' in name or \
                        'root_embed' in name or \
                        'body_embed' in name or \
                        'trans_out' in name or \
                        'root_out' in name or \
                        'body_out' in name:
                        param.requires_grad = requires_grad
    elif mode=='root_face':
        # only support to unfreeze root and trans and face encoder and decoder
        for net in nets:
            if net is not None:
                for name, param in net.named_parameters():
                    if 'trans_embed' in name or \
                        'root_embed' in name or \
                        'body_embed' in name or \
                        'trans_out' in name or \
                        'root_out' in name or \
                        'body_out' in name or \
                        'face_embed' in name or \
                        'face_out' in name:
                        param.requires_grad = requires_grad
    elif mode=='root_hand':
        # only support to unfreeze root and trans and hands encoder and decoder
        for net in nets:
            if net is not None:
                for name, param in net.named_parameters():
                    if 'trans_embed' in name or \
                        'root_embed' in name or \
                        'body_embed' in name or \
                        'trans_out' in name or \
                        'root_out' in name or \
                        'body_out' in name or \
                        'lhand_embed' in name or \
                        'rhand_embed' in name or \
                        'lhand_out' in name or \
                        'rhand_out' in name:
                        param.requires_grad = requires_grad
    elif mode=='root_face_hand':
        # only support to unfreeze root and trans and face and hands encoder and decoder
        for net in nets:
            if net is not None:
                for name, param in net.named_parameters():
                    if 'trans_embed' in name or \
                        'root_embed' in name or \
                        'body_embed' in name or \
                        'trans_out' in name or \
                        'root_out' in name or \
                        'body_out' in name or \
                        'face_embed' in name or \
                        'face_out' in name or \
                        'lhand_embed' in name or \
                        'rhand_embed' in name or \
                        'lhand_out' in name or \
                        'rhand_out' in name:
                        param.requires_grad = requires_grad
    else:
        raise NotImplementedError

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
