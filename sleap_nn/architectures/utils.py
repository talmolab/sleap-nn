"""Miscellaneous utility functions for architectures and modeling."""

import torch
from torch import nn
from loguru import logger


def get_act_fn(activation: str) -> nn.Module:
    """Get an instance of an activation function module based on the provided name.

    This function returns an instance of a PyTorch activation function module
    corresponding to the given activation function name.

    Args:
        activation (str): Name of the activation function. Supported values are 'relu', 'sigmoid', 'tanh', 'softmax', and 'identity'.

    Returns:
        nn.Module: An instance of the requested activation function module.

    Raises:
        KeyError: If the provided activation function name is not one of the supported values.

    Example:
        # Get an instance of the ReLU activation function
        relu_fn = get_act_fn('relu')

        # Apply the activation function to an input tensor
        input_tensor = torch.randn(1, 64, 64)
        output = relu_fn(input_tensor)
    """
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=-1),
        "identity": nn.Identity(),
    }

    if activation not in activations:
        message = f"Unsupported activation function: {activation}. Supported activations are: {', '.join(activations.keys())}"
        logger.error(message)
        raise KeyError(message)

    return activations[activation]


def get_children_layers(model: torch.nn.Module):
    """Recursively retrieves a flattened list of all children modules and submodules within the given model.

    Args:
        model: The PyTorch model to extract children from.

    Returns:
        list of nn.Module: A flattened list containing all children modules and submodules.
    """
    children = list(model.children())
    flattened_children = []
    if len(children) == 0:
        return model
    else:
        for child in children:
            try:
                flattened_children.extend(get_children_layers(child))
            except TypeError:
                flattened_children.append(get_children_layers(child))
    return flattened_children
