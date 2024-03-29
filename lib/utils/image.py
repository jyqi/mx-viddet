# --------------------------------------------------------
# Accel
# Copyright (c) 2019
# Licensed under the MIT License [see LICENSE for details]
# Modified by Samvit Jain
# --------------------------------------------------------

import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes


# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def get_ref_imgs(seg_rec, config, num):

    eq_flag = 0 # 0 for unequal, 1 for equal

    prefix = ''
    suffix = ''
    if seg_rec['flipped']:
        prefix = seg_rec['image'][:-len('000019_leftImg8bit_flip.png')]
        suffix = seg_rec['image'][-len('000019_leftImg8bit_flip.png'):]
    else:
        prefix = seg_rec['image'][:-len('000019_leftImg8bit.png')]
        suffix = seg_rec['image'][-len('000019_leftImg8bit.png'):]

    frame_id = int(suffix[:len('000019')])

    ref_imgs = []
    for idx in range(num, 0, -1):

        ref_id = frame_id - idx
        # print 'frame id: {}\n ref id: {}'.format(frame_id, ref_id)

        ref_prefix = prefix[len('./data/cityscapes/leftImg8bit/'):]
        ref_im_name = '/city/leftImg8bit_sequence/' + ref_prefix + ('%06d' % ref_id) + '_leftImg8bit.png'
        # print seg_rec['image']
        # print ref_im_name

        # read ref image
        if not os.path.exists(ref_im_name):
            print '{} does not exist'.format(ref_im_name)
            ref_id = frame_id
            ref_im_name = seg_rec['image']
        ref_im = np.array(cv2.imread(ref_im_name))

        ref_imgs.append(ref_im)

    return ref_imgs, eq_flag

def get_segmentation_pair(segdb, config):
    """
    propocess image and return segdb
    :param segdb: a list of segdb
    :return: list of img as mxnet format
    """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_segdb = []
    processed_seg_cls_gt = []
    for i in range(num_images):
        seg_rec = segdb[i]

        assert os.path.exists(seg_rec['image']), '%s does not exist'.format(seg_rec['image'])
        im = np.array(cv2.imread(seg_rec['image']))

        assert not seg_rec['flipped']
        ref_imgs, eq_flag = get_ref_imgs(seg_rec, config, config.TRAIN.KEY_INTERVAL - 1)

        new_rec = seg_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensors = []
        for ref_im in ref_imgs:
            ref_im, ref_im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            tensor = transform(ref_im, config.network.PIXEL_MEANS)
            ref_im_tensors.append(tensor)
        # print ref_im_tensors[0].shape
        ref_im_tensor = np.concatenate(ref_im_tensors, axis=0)
        # print ref_im_tensor.shape
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info

        seg_cls_gt = np.array(Image.open(seg_rec['seg_cls_path']))
        seg_cls_gt, seg_cls_gt_scale = resize(
            seg_cls_gt, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=cv2.INTER_NEAREST)
        seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)

        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        processed_segdb.append(new_rec)
        processed_seg_cls_gt.append(seg_cls_gt_tensor)

    return processed_ims, processed_ref_ims, processed_eq_flags, processed_seg_cls_gt, processed_segdb

def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb


def get_triple_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_bef_ims = []
    processed_aft_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            # get two different frames from the interval [frame_id + MIN_OFFSET, frame_id + MAX_OFFSET]
            offsets = np.random.choice(config.TRAIN.MAX_OFFSET - config.TRAIN.MIN_OFFSET + 1, 2, replace=False) + config.TRAIN.MIN_OFFSET
            bef_id = min(max(roi_rec['frame_seg_id'] + offsets[0], 0), roi_rec['frame_seg_len']-1)
            aft_id = min(max(roi_rec['frame_seg_id'] + offsets[1], 0), roi_rec['frame_seg_len']-1)
            bef_image = roi_rec['pattern'] % bef_id
            aft_image = roi_rec['pattern'] % aft_id

            assert os.path.exists(bef_image), '%s does not exist'.format(bef_image)
            assert os.path.exists(aft_image), '%s does not exist'.format(aft_image)
            bef_im = cv2.imread(bef_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            aft_im = cv2.imread(aft_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            bef_im = im.copy()
            aft_im = im.copy()

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            bef_im = bef_im[:, ::-1, :]
            aft_im = aft_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        bef_im, im_scale = resize(bef_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        aft_im, im_scale = resize(aft_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        bef_im_tensor = transform(bef_im, config.network.PIXEL_MEANS)
        aft_im_tensor = transform(aft_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_bef_ims.append(bef_im_tensor)
        processed_aft_ims.append(aft_im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
        with open('log3.txt', 'a') as f:
            f.write("%s\t%s\n" % (roi_rec['image'], str(im_info)))
    return processed_ims, processed_bef_ims, processed_aft_ims, processed_roidb


def get_segmentation_image(segdb, config):
    """
    propocess image and return segdb
    :param segdb: a list of segdb
    :return: list of img as mxnet format
    """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_ims = []
    processed_segdb = []
    processed_seg_cls_gt = []
    for i in range(num_images):
        seg_rec = segdb[i]
        assert os.path.exists(seg_rec['image']), '%s does not exist'.format(seg_rec['image'])
        im = np.array(cv2.imread(seg_rec['image']))

        new_rec = seg_rec.copy()

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info

        seg_cls_gt = np.array(Image.open(seg_rec['seg_cls_path']))
        seg_cls_gt, seg_cls_gt_scale = resize(
            seg_cls_gt, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=cv2.INTER_NEAREST)
        seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)

        processed_ims.append(im_tensor)
        processed_segdb.append(new_rec)
        processed_seg_cls_gt.append(seg_cls_gt_tensor)

    return processed_ims, processed_seg_cls_gt, processed_segdb

def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
