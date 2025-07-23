import yaml
import functools
import torch.nn as nn
import torch
import platform
from torchviz import make_dot
import os
import tempfile
import pandas as pd
from PIL import Image
import numpy as np
from scipy.ndimage import center_of_mass
from torchvision import transforms
import colorsys

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def get_config(config):
    with open(config, 'r') as stream:
        config_dict = yaml.load(stream,Loader=yaml.SafeLoader)
    return Config(config_dict)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
    
class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        
class Identity(nn.Module):
    def forward(self, x):
        return x
    
def log_model_params(logger, config):
    config_attributes = [attr for attr in dir(config) if not attr.startswith('__')]
    config_dict = {}
    for attr in config_attributes:
            value = getattr(config, attr)
            config_dict[attr] = value
    config_df = pd.DataFrame([config_dict])
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
        config_df.to_csv(temp.name, index=False)
        logger["config"].upload(temp.name)
        temp.close()
    
def get_device():
    # check if GPU is available
    num_gpus = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        if num_gpus > 1:
            return torch.device('cuda:0')
        else:
            return torch.device('cuda')
    
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    
    else:
        if 'mac' in platform.system().lower():
                print("Consider setting up your macbook to use Metal Performance Shaders if using Apple Silicon")
        return torch.device('cpu')
    
def log_model_viz(model_name, model_save_dir, x, model, logger, show_attrs=False, show_saved=False):
    vis = make_dot(model(x), params=dict(model.named_parameters()), show_attrs=show_attrs, show_saved=show_saved)
    vis_path = os.path.join(model_save_dir, f'{model_name}_vis')
    vis.format = 'png'
    vis.render(vis_path)
    logger[f'{model_name}_vis'].append(Image.open(f'{vis_path}.png'))
    
def make_z_gradient(shape):
    height, width = shape[:2]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            red = int((x / width) * 255)
            blue = int((y / height) * 255)
            green = int(((x + y) / (width + height)) * 255)
            image[y, x] = (red, green, blue)
    
    return image
"""
def colorize_labels(labels, z):
    color_sums = np.zeros((labels.max() + 1, z.shape[2]))
    np.add.at(color_sums, labels.ravel(), z.reshape(-1, z.shape[2]))
    counts = np.bincount(labels.ravel())
    average_colors = color_sums / counts[:, None]
    colored_labels = np.where(labels[:, :, None] == 0, labels[:, :, None], average_colors[labels])
    colored_labels = (normalize(colored_labels) * 255).astype(np.uint8)
    return colored_labels"""

def semantic_binary_map(labels):
    return (labels > 0).astype(int)

def rgb_to_labels(colored_labels):
    flat_pixels = list(map(tuple, colored_labels.reshape(-1, 3)))
    non_black_pixels = [pixel for pixel in flat_pixels if pixel != (0, 0, 0)]
    unique_colors = {color: i + 1 for i, color in enumerate(set(non_black_pixels))}
    labels = np.array([unique_colors.get(pixel, 0) for pixel in flat_pixels])
    labels = labels.reshape(colored_labels.shape[:2])
    return labels

def get_label_centroids(labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    centroids = np.empty((len(unique_labels), 2), dtype=int)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroid = center_of_mass(mask)
        centroids[i] = [int(centroid[1]), int(centroid[0])]  # Swap the order to get (x, y)

    return centroids

def get_num_instances(labels):
    return len(torch.unique(labels)) - 1

def rgb_to_num_instances(input_labels):
    labels = input_labels.clone()
    black = torch.zeros_like(labels[0])

    for im in range(labels.shape[0]):
        # Create a mask of non-black pixels
        non_black_mask = (labels[im] != black).all(dim=-1)

        # Get unique colors and their inverse mapping
        unique_colors = torch.unique(labels[im][non_black_mask], dim=0)
        inverse = torch.empty_like(labels[im][non_black_mask])
        for i, color in enumerate(unique_colors):
            mask = (labels[im][non_black_mask] == color).all(dim=-1)
            inverse[mask] = i + 1

        # Assign each unique color a different label
        labels[im][non_black_mask] = inverse
    num_instances = get_num_instances(labels)

    return num_instances

def rgb_to_labels_centroids(torch_labels, return_labeled=False):
    device = torch_labels.device
    labels = torch_labels.cpu().numpy()
    height, width = labels.shape[-2:]

    for im in range(labels.shape[0]):
        # Create a mask of non-black pixels
        non_black_mask = (labels[im] != [0, 0, 0]).all(axis=-1)

        # Get unique colors and their inverse mapping
        unique_colors, inverse = np.unique(labels[im][non_black_mask], axis=0, return_inverse=True)

        # Assign each unique color a different label
        labels[im][non_black_mask] = inverse + 1

    # Normalize the centroids
    centroids = get_label_centroids(labels).astype(float)
    centroids /= [width, height]
    centroids = torch.from_numpy(centroids).to(device)

    if return_labeled:
        return labels, centroids
    return centroids

def process_im_for_log(arr):
    arr = arr / np.max(arr)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0))
    return arr

def generate_colors_hsl(n):
    colors = [(0, 0, 0)]  # start with black
    for i in range(1, n):  # adjust range to be from 1 to n
        hue = i / n
        lightness = 0.5  # or any other constant value
        saturation = 0.9  # or any other constant value
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors

def colorize_labels(labels, colors):
    # Create an empty image
    image = np.zeros((labels.shape[0], labels.shape[1], 3))
    unique_labels = np.unique(labels)
    for n, i in enumerate(unique_labels):
        if i == 0:
            continue
        color_index = int(n / len(unique_labels) * len(colors))
        image[labels == i] = colors[color_index]
    image = (image * 255).astype(np.uint8) 
    return image

def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Denormalize (values will be between 0 and 1)
    tensor = tensor * 255  # Scale pixel values to 0-255
    if tensor.shape[0] == 1:
        tensor = tensor[0, :, :]
    return tensor

def make_z_color(grid,alpha=1.0):
    # Assuming z is a numpy.ndarray variable
    z = np.random.choice([0, 1], size=grid.shape[0:2], p=[0.95, 0.05])
    #z = np.random.normal(0.1, 0.1, size=grid.shape[0:2])
    z = np.stack([z, z, z], axis=-1)
    z = z * grid
    z = (z*255*alpha).astype(np.uint8)
    return z