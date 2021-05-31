"""
Model related utils.
"""


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print(f"Number of parameters of {name}: {num_params}")
