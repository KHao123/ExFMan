import torch
import math
import random
import numpy as np

img2mse = lambda x, y : torch.mean((x - y) ** 2)


def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()


def event_loss_call(all_rgb, event_data, rgb2gray="ave"):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    if rgb2gray == "rgb":
        rgb2grey = torch.tensor([0.299,0.587,0.114]).cuda()
    elif rgb2gray == "ave":
        rgb2grey = torch.tensor([1/3, 1/3, 1/3]).cuda()
    

    event_rgb_g = all_rgb[1].mean(dim=-1, keepdim=True)
    rgb_g = all_rgb[0].mean(dim=-1, keepdim=True)


    SCALE = 2.2 
    THR = 0.5
    EPS = 1e-5 
    RATIO = SCALE/THR
    pred_event_1 = torch.log(event_rgb_g**SCALE+EPS)-torch.log(rgb_g.detach()**SCALE+EPS)
    pred_event_2 = torch.log(event_rgb_g.detach()**SCALE+EPS)-torch.log(rgb_g**SCALE+EPS)

    event_loss = (img2mse(pred_event_1, event_data*THR) + img2mse(pred_event_2, event_data*THR)) / 2

    return event_loss

def get_event_rgb(event_map):
    truth_event_rgb = np.zeros_like(event_map)
    mask = event_map[:,:,0] > 0
    truth_event_rgb[mask, :] = (event_map[mask, :] / 10) * np.array([255, 0, 0])
    mask = event_map[:,:,0] < 0
    truth_event_rgb[mask, :] = (event_map[mask, :] / 10) * np.array([0, -255, 0])
    truth_event_rgb = truth_event_rgb.astype(np.uint8)

    return truth_event_rgb