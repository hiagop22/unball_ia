import json
import torch
import torch.onnx
import numpy as np
from torch import nn
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        return json.dump(data, f, indent=4)

def save_onnx(model, save_path, state_size: int):
    try: 
        conv2onnx(model, Path(save_path) / 'last-model.onnx', size=(state_size, 1))
    except IndexError:
        print('Best model not found, skipping torch save')
    

def conv2onnx(model, outpath, size):
    
    w,h = size
    model.eval()
    device = torch.device('cuda:1')
    x = torch.randn((h, w)).to(device)

    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        outpath,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable lenght axes
            "output": {0: "batch_size"},
        },
    )


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' in rads. 
    The angle can be positive or negative
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_v1, angle_v2 = np.arctan2([v1_u[1],v2_u[1]], [v1_u[0], v2_u[0]])
    
    delta = angle_v1 - angle_v2
    
    if delta > np.pi:
        delta = delta - 2*np.pi
    elif delta < -np.pi:
        delta = 2*np.pi + delta
    
    return -delta

def normalized_angle(v1, v2):
    # Receive 2 vectors (the order is important) and return the the range (-1,1)
    return angle_between(v1,v2)/np.pi


def norm_grad(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def weights_init(m):
    # for every Linear layer in a model..
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)