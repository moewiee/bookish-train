import torch
import numpy as np

def to_tensor(image):
    image = torch.from_numpy(np.asarray(image)).float()
    image.unsqueeze_(0).div_(255.)
    return image