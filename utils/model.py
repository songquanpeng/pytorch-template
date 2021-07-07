"""
Model related utils.
"""


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print(f"Parameter number of {name}: {num_params/1e6:.4f}M")
