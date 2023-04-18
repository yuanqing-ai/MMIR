import torch
from torchvision import transforms
""" Cropping functions to apply the same crop to all frames in a video segment
"""

def _crop_segment(segment, o_h, crop_height, o_w, crop_width, input_size):
    return segment[:, o_h:(o_h + crop_height), o_w:(o_w + crop_width)]


def _training_offsets(crop_pairs, width, height,h_rand,w_rand):
    w_step = (width - crop_pairs[0][1]) / 4
    
    #w_step = torch.tensor(w_step,dtype=torch.int32)
    #print("w_step",w_step.dtype)
    h_step = (height - crop_pairs[0][0]) / 4
    #h_step.type(torch.int32)
    o_h = torch.mul(h_rand, h_step)
    o_w = torch.mul(w_rand, w_step)
    #o_h.type(torch.int32)
    #o_w.type(torch.int32)

    return o_h, o_w


def training_crop(segment, height, width, input_size, scales, s_rand, h_rand, w_rand):
    #print("scales",scales)
    crop_pairs = scales[s_rand, :]
    #print('crop_pairs[0][1]',crop_pairs[0][1].dtype)
    #print("segment",segment)
    # Choose random crop given scale
    o_h, o_w = _training_offsets(crop_pairs, width, height, h_rand, w_rand)
    o_h = int(o_h.numpy())
    o_w = int(o_w.numpy())

    
    # Crop segment
    return _crop_segment(segment, o_h, crop_pairs[0][0], o_w, crop_pairs[0][1], input_size)


def testing_crop(segment, height, width, input_size):
    o_w = (width - input_size) / 2
    #o_w.type(torch.int32)
    o_h = (height - input_size) / 2
    o_h = int(o_h)
    o_w = int(o_w)
    #o_h.type(torch.int32)
    return segment[:, o_h:(o_h + input_size), o_w:(o_w + input_size)]


def random_flip(segment, f_rand, invert=False):
    # Choose random number
    mirror_cond = torch.tensor(bool(f_rand<0.5),dtype=bool)

    
    invert_cond = invert and mirror_cond

    invert_cond = torch.as_tensor(invert_cond,dtype=bool)
    segment = torch.where(mirror_cond, torch.flip(segment, dims=[-2]), segment)
    inverted_segment = 255 - segment
    return torch.where(invert_cond, inverted_segment, segment)


def normalise(segment,flow):
    segment = segment / 255.0
    # segment = segment - 0.5
    # return segment * 2.0

    if flow:
        #print("segment",segment.size())
        segment = transforms.Normalize(mean=[0.5, 0.5],
                             std=[0.5, 0.5])(segment)

    else:
        #print("segment",segment.size())

        segment = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(segment)

    return segment
