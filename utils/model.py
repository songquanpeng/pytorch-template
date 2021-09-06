"""
Model related utils.
"""


def count_parameters(network, name):
    num_params = 0
    num_trainable_params = 0
    for p in network.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_trainable_params += p.numel()
    print(f"Parameter number of {name}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable)")
