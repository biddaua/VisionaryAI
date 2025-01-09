
"""
 build optimizer for ms
"""
from mindspore.nn.optim.adam import Adam, AdamWeightDecay


def build_optimizer(model, opts, lr, enable_lora=False):
    """

    :param model:
    :param opts:
    :param lr:
    :return: optimizer
    """

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    param_optimizer = model.trainable_params()
    decay_params = list(filter(decay_filter, param_optimizer))
    other_params = list(filter(lambda x: not decay_filter(x), param_optimizer))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-6
    }, {
            'order_params': param_optimizer
    }]

    # 适配lora后，得到的other_params为空，因此无需加入到group_params中
    if not enable_lora:
        group_params.append({
            'params': other_params,
            'weight_decay': 0.0
        })

    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamw':
        OptimCls = AdamWeightDecay
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(group_params,
                         learning_rate=lr, beta1=opts.betas[0], beta2=opts.betas[1])
    return optimizer
