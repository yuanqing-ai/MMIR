import torch
import numpy as np
from data_gen.crops import *
from PIL import Image
import torch.nn.functional as F


class DataAugmentation:

    """ Reads lists of frames of a video segmenti nto a Tensor. Applies TSN style augmentations.
        Limits input frames to be of the same size defined by original_height and original_width.

        (input_size) Hight and Width of augmented frame
        (scales) List of possible scaling factors to scale width and hight
        (original_height) Height of the input frames
        (original_width) Width of the input frames
        (flow) True: Process u and v optical flow frames. False: RGB frames
        (summary): Produce image summaries to tensorboard

        To run augmentation call preprocess method:
        (filenames) filenames of frames in a single video segment
        (is_training) don't augment and use center crop

        To apply same augmentation to rgb and flow call preprocess_rgb_flow
        (filenames) filenames of frames in a single video segment
        (is_training) don't augment and use center crop
    """

    def __init__(self, input_size, scales, original_height, original_width, flow, summary=False):
        def _get_crop_list(pre_h, pre_w):
            base_size = min(pre_h, pre_w)
            crop_sizes = [int(base_size * x) for x in self.scales]
            crop_sizes = [self.input_size if abs(x - self.input_size) < 3 else x for x in crop_sizes]
            pairs = []
            for i, h in enumerate(crop_sizes):
                for j, w in enumerate(crop_sizes):
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))
            return pairs

        self.summary=summary
        self.input_size = input_size
        self.height = original_height
        self.width = original_width
        self.max_distort = 1
        self.scales = scales
        self.crops = np.array(_get_crop_list(self.height, self.width), )
        self.num_scales = self.crops.shape[0]
        self.flow = flow

    def _training_preprocess(self, segment, modality,f_rand,  s_rand, h_rand, w_rand):
        scales = torch.from_numpy(self.crops)
        segment = training_crop(segment, self.height, self.width, self.input_size, scales,  s_rand, h_rand, w_rand)
        if modality == "u":
            return random_flip(segment, f_rand=f_rand, invert=True)
        elif modality == "v":
            return random_flip(segment,f_rand=f_rand)
        elif modality == "RGB":
            return random_flip(segment,f_rand=f_rand)
        else:
            raise Exception("Unknown modality type for random flipping")

    def _test_preprocess(self, segment):
        return testing_crop(segment, self.height, self.width, self.input_size)

    def _preprocess_fn(self, filename, is_training, f_rand, h_rand, w_rand, s_rand, name="Modality name"):

        if name=="RGB":
            num_channel = 'RGB'
        elif name=="u" or name=="v":
            num_channel = 'L'
        else:
            raise Exception("Could not specify number of channels, unkown modality: "+name)

        # segment = tf.cast(tf.map_fn(lambda f: tf.image.decode_jpeg(tf.read_file(f), channels=num_channel),
        #                             filenames, dtype=tf.uint8), dtype=tf.float32)
        # segment_aug = tf.cond(is_training, lambda: self._training_preprocess(segment, name, f_rand,  s_rand, h_rand, w_rand),
        #                lambda: self._test_preprocess(segment))

        segment = filename
        #print("before",segment.shape)

        if is_training:
            segment_aug = self._training_preprocess(segment, name, f_rand, s_rand, h_rand, w_rand).permute(0, 3, 1, 2)
            #print("segment_aug",segment_aug.shape)
        else:
            segment_aug = self._test_preprocess(segment).permute(0, 3, 1, 2)
            #print("segment_aug",segment_aug.shape)


        resized_segment_aug = F.interpolate(segment_aug, size=(self.input_size, self.input_size), mode='bilinear', align_corners=True)

        #print(resized_segment_aug.shape)

        return resized_segment_aug

    def preprocess_image(self, filenames, is_training):
        # f_rand = tf.random_uniform([], 0.0, 1.0)
        # h_rand = tf.multiply(tf.random_uniform([], 0, 2, dtype=tf.int32), 2)
        # w_rand = tf.multiply(tf.random_uniform([], 0, 2, dtype=tf.int32), 2)
        # s_rand = tf.random_uniform([], 0, self.num_scales, dtype=tf.int32)

        f_rand = torch.rand(1)
        h_rand = torch.randint(low=0, high=2, size=(1,)) * 2
        w_rand = torch.randint(low=0, high=2, size=(1,)) * 2
        s_rand = torch.randint(low=0, high=self.num_scales, size=(1,))

        return self._preprocess_fn(filenames, is_training, f_rand, h_rand, w_rand, s_rand, name="RGB")

    def preprocess_flow_correct(self, filenames, is_training):
        f_rand = torch.rand(1)
        h_rand = torch.randint(low=0, high=2, size=(1,)) * 2
        w_rand = torch.randint(low=0, high=2, size=(1,)) * 2
        s_rand = torch.randint(low=0, high=self.num_scales, size=(1,))
        # segment = filenames
        # # filenames_u = filenames[:, 0]
        # # filenames_v = filenames[:, 1]
        segment = self._preprocess_fn(filenames, is_training, f_rand, h_rand, w_rand, s_rand, name="u")
        # segment_v = self._preprocess_fn(filenames_v, is_training, f_rand, h_rand, w_rand, s_rand, name="v")
        # segment = tf.concat([segment_u, segment_v], axis=-1)
        return segment

    def preprocess_rgb_flow(self, filenames, is_training):
        f_rand = tf.random_uniform([], 0.0, 1.0)
        h_rand = tf.multiply(tf.random_uniform([], 0, 2, dtype=tf.int32), 2)
        w_rand = tf.multiply(tf.random_uniform([], 0, 2, dtype=tf.int32), 2)
        s_rand = tf.random_uniform([], 0, self.num_scales, dtype=tf.int32)
        filenames_RGB = filenames[:, 0]
        filenames_u = filenames[:, 1]
        filenames_v = filenames[:, 2]
        segment_rgb = self._preprocess_fn(filenames_RGB, is_training, f_rand, h_rand, w_rand, s_rand, name="RGB")
        segment_u = self._preprocess_fn(filenames_u, is_training, f_rand, h_rand, w_rand, s_rand, name="u")
        segment_v = self._preprocess_fn(filenames_v, is_training, f_rand, h_rand, w_rand, s_rand, name="v")
        segment_flow = torch.cat([segment_u, segment_v], dim=-1)

        return normalise(segment_rgb,False), normalise(segment_flow,True)


    def preprocess(self, filenames, is_training):
        if self.flow:
            segment_aug = self.preprocess_flow_correct(filenames, is_training)
        else:
            segment_aug = self.preprocess_image(filenames, is_training)

        return normalise(segment_aug,self.flow)
