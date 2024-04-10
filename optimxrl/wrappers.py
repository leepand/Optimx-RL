"""
A collection of objects thats can wrap / otherwise modify arbitrary neural
network layers.
"""


from .ActivationFunctions.ActivationLayers import (
    SigmoidLayer,
    ReLU,
    Tanh,
    Swish,
    LeakyReLU,
    Iden,
)


def get_activation_layer_function(out_type="sig"):
    if out_type == "sig":
        OutLayer = SigmoidLayer
    elif out_type == "relu":
        OutLayer = ReLU
    elif out_type == "tanh":
        OutLayer = Tanh
    elif out_type == "swish":
        OutLayer = Swish
    elif out_type == "leak":
        OutLayer = LeakyReLU
    elif out_type == "iden":
        OutLayer = Iden
    else:
        raise ValueError(f"{out_type} is not supported.")

    return OutLayer
