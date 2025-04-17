import time
import torch
import math
###############################################################################
## Misc Functions
###############################################################################

def cpu_data_to_gpu(cpu_data, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    gpu_data = {}
    for key, val in cpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                gpu_data[key] = [x.cuda() for x in val]
        elif isinstance(val, dict):
            gpu_data[key] = {sub_k: sub_val.cuda() for sub_k, sub_val in val.items()}
        else:
            gpu_data[key] = val.cuda()

    return gpu_data


###############################################################################
## Timer
###############################################################################

class Timer():
    def __init__(self):
        self.curr_time = 0

    def begin(self):
        self.curr_time = time.time()

    def log(self):
        diff_time = time.time() - self.curr_time
        self.begin()
        return f"{diff_time:.2f} sec"

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x*f, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = 1e8
    y = torch.round(y*rounding)/rounding

    return y.float()

def event_loss_call(all_rgb, event_data, rgb2gray):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    if rgb2gray == "rgb":
        rgb2grey = torch.tensor([0.299,0.587,0.114]).to(event_data.device)
    elif rgb2gray == "ave":
        rgb2grey = torch.tensor([1/3, 1/3, 1/3]).to(event_data.device)
    loss = 0

    thres_pos = (lin_log(torch.mv(all_rgb[1], rgb2grey) * 255) - lin_log(torch.mv(all_rgb[0], rgb2grey) * 255)) / 0.2
    thres_neg = (lin_log(torch.mv(all_rgb[1], rgb2grey) * 255) - lin_log(torch.mv(all_rgb[0], rgb2grey) * 255)) / 0.2

    pos = event_data > 0
    neg = event_data < 0

    loss_pos = torch.mean(((thres_pos * pos) - ((event_data + 0.5) * pos)) ** 2)
    loss_neg = torch.mean(((thres_neg * neg) - ((event_data - 0.5) * neg)) ** 2)

    loss += loss_pos + loss_neg

    return loss

def event_loss_call_uncert(all_rgb, event_data, rgb2gray, uncert):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    if rgb2gray == "rgb":
        rgb2grey = torch.tensor([0.299,0.587,0.114]).to(event_data.device)
    elif rgb2gray == "ave":
        rgb2grey = torch.tensor([1/3, 1/3, 1/3]).to(event_data.device)
    loss = 0
    
    pos = event_data > 0
    neg = event_data < 0

    ratio = uncert[0] / uncert[1]
    mask_1 = ratio < 1
    mask_2 = ratio > 1

    # mask_1
    thres_pos = (lin_log(torch.mv(all_rgb[1], rgb2grey) * 255) - lin_log(torch.mv(all_rgb[0].detach(), rgb2grey) * 255)) / 0.2
    thres_neg = (lin_log(torch.mv(all_rgb[1], rgb2grey) * 255) - lin_log(torch.mv(all_rgb[0].detach(), rgb2grey) * 255)) / 0.2

    loss_pos = torch.mean(((thres_pos * pos) - ((event_data + 0.5) * pos)) ** 2 * mask_1 / ratio)
    loss_neg = torch.mean(((thres_neg * neg) - ((event_data - 0.5) * neg)) ** 2 * mask_1 / ratio)

    loss += loss_pos + loss_neg

    # mask_2
    thres_pos = (lin_log(torch.mv(all_rgb[1].detach(), rgb2grey) * 255) - lin_log(torch.mv(all_rgb[0], rgb2grey) * 255)) / 0.2
    thres_neg = (lin_log(torch.mv(all_rgb[1].detach(), rgb2grey) * 255) - lin_log(torch.mv(all_rgb[0], rgb2grey) * 255)) / 0.2
    
    loss_pos = torch.mean(((thres_pos * pos) - ((event_data + 0.5) * pos)) ** 2 * mask_2 * ratio)
    loss_neg = torch.mean(((thres_neg * neg) - ((event_data - 0.5) * neg)) ** 2 * mask_2 * ratio)

    loss += loss_pos + loss_neg

    return loss