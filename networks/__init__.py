from networks.caffenet import caffenet
from networks.mnist import lenet
from networks.resnet import resnet18, resnet50, resnet50_fc, resnet18_fc
from networks.alexnet import alexnet
from networks.advnet import advnet
from networks.random_layer import random_layer

nets_map = {
    'caffenet': caffenet,
    'alexnet': alexnet,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet50_fc': resnet50_fc,
    'resnet18_fc': resnet18_fc,
    'lenet': lenet,
    'advnet': advnet,
    'random_layer': random_layer
}


def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
