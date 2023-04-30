import torch
import numpy as np

def make_displayable(img, channels_dim=0, alpha=0.8):
    """ Make segmentation image (multichannel) : size = (C, H, W)

    Args:
        img (_type_): (C, H, W) image
        channels_dim  (int, optional): the channels dimension. Defaults to 0.
        alpha (float, optional): transparency effect on classes
        (usefull for overlaping classes). Defaults to 0.8.

    Returns:
       a (H, W) image with colors for each classes
    """
    _, merged = torch.max(img, dim=channels_dim)
    colored = np.zeros((merged.shape[0], merged.shape[1], 4), dtype=np.float32) # 4 channels to have alpha

    colors = [(1,0,0, alpha), (0,1,0, alpha), (0,0,1, alpha), (1,0,1, alpha),
               (1,1,0, alpha), (0,1,1, alpha), (0.2, 0, 0.8, alpha), (0.8, 0, 0.2, alpha),
               (0, 0.2, 0.8, alpha), (0, 0.8, 0.2, alpha)]
    bg_color = (0,0,0, 1.)
    nb_class = img.shape[channels_dim]
    if nb_class > len(colors) :
        print("WARNING:make_displayable: Some classes will have the same color")
    for i in range(1, nb_class) :
        colored[merged == i] = colors[i]
    colored[merged == 0] = bg_color
    return colored

def visualize(img):
    _min = img.min()
    _max = img.max()
    if np.isclose(_min, 0) and np.isclose(_max, 0) :
        return img
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def merge_classes(img, channels_dim=0) :
    """
    Args:
        img (ndarray): multichannels image (C, H, W)
    """
    return torch.max(img, dim=channels_dim)[1]