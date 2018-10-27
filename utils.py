#!/usr/bin/env python2.7

import numpy as np
import cv2
from keras import backend as K
import os
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_SAX_SERIES():
    SAX_SERIES = {}
    with open('SAX_series.txt', 'r') as f:
        for line in f:
            if not line.startswith('#'):
                key, val = line.split(':')
                SAX_SERIES[key.strip()] = val.strip()

    return SAX_SERIES


def mvn(ndarray):
    '''Input ndarray is of rank 3 (height, width, depth).

    MVN performs per channel mean-variance normalization.
    '''
    epsilon = 1e-6
    mean = ndarray.mean(axis=(0,1), keepdims=True)
    std = ndarray.std(axis=(0,1), keepdims=True)

    return (ndarray - mean) / (std + epsilon)


def reshape(ndarray, to_shape):
    '''Reshapes a center cropped (or padded) array back to its original shape.'''
    h_in, w_in, d_in = ndarray.shape
    h_out, w_out, d_out = to_shape
    if h_in > h_out: # center crop along h dimension
        h_offset = (h_in - h_out) / 2
        ndarray = ndarray[h_offset:(h_offset+h_out), :, :]
    else: # zero pad along h dimension
        pad_h = (h_out - h_in)
        rem = pad_h % 2
        pad_dim_h = (pad_h/2, pad_h/2 + rem)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, (0,0), (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
    if w_in > w_out: # center crop along w dimension
        w_offset = (w_in - w_out) / 2
        ndarray = ndarray[:, w_offset:(w_offset+w_out), :]
    else: # zero pad along w dimension
        pad_w = (w_out - w_in)
        rem = pad_w % 2
        pad_dim_w = (pad_w/2, pad_w/2 + rem)
        npad = ((0,0), pad_dim_w, (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
    
    return ndarray # reshaped


def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    '''
    h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')
    
    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h/2, pad_h/2 + rem_h)
        pad_dim_w = (pad_w/2, pad_w/2 + rem_w)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w, d = ndarray.shape
    # center crop
    h_offset = (h - crop_size) / 2
    w_offset = (w - crop_size) / 2
    cropped = ndarray[h_offset:(h_offset+crop_size),
                      w_offset:(w_offset+crop_size), :]

    return cropped


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter)))**power
    K.set_value(model.optimizer.lr, lrate)

    return K.eval(model.optimizer.lr)


def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=None)
    summation = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None)
    
    return 2.0 * intersection / summation


def jaccard_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=None)
    union = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None) - intersection

    return float(intersection) / float(union)


def get_confusion_matrix_bbox(mask, y_pred):
    '''
        Using confusion matrix to evaluate the performance of cropping
        For each mask - pred pair, compute the bbox of pred, regard mask as ground truth, bbox as prediction,
        apply confusion matrix metrics. After that, average over all confusion matrix.
    '''
    pred_box = np.zeros((mask.shape))
    n = mask.shape[0]
    for i in range(n):
        pred = y_pred[i, 0, :,:]
        [x_min, x_max, y_min, y_max] = get_bbox_single(pred)
        pred_box[i,  x_min:x_max, y_min:y_max, 0] = 1
    pred_box = np.reshape(pred_box, [n, pred_box.shape[1]*pred_box.shape[1]])
    mask = np.reshape(mask, [n, mask.shape[1] * mask.shape[1]])
    #cm = confusion_matrix(mask, pred_box)
    cm = np.zeros((2,2))
    for i in range(n):
        cm = cm + confusion_matrix(mask[i,:], pred_box[i,:])
    cm = cm / n
    return cm

def get_cropped(img, y_pred, roi_size = 32, win_size = 100):
    '''
        Cropped the original image using CNN prediction
        @param:
            img: the original image, shape (N, WIDTH, HEIGHT, 1), default size 256
            y_pred: the prediction of ROI, may be showed as scatter binary image, shape (N, 1, roi_size, roi_size)
            roi_size: the size of y_pred, default 32
            win_size: the size of window used to crop the original image, default 80
        @return
            cropped: the cropped image, same format with input img, but with smaller size of win_size
    '''
    n = img.shape[0]
    cropped = np.zeros((n, win_size, win_size, 1))
    for i in range(y_pred.shape[0]):
        pred = y_pred[i, 0, :,:]
        [x_min, x_max, y_min, y_max] = get_bbox_single(pred, win_size = win_size)
        cropped[i, :, :, 0] = img[i, x_min:x_max, y_min:y_max, 0]
    return cropped

def get_bbox_single(pred, roi_size = 32, win_size = 100):
    '''
        Compute the bounding box param of the given binary region mask
        This implementation compute the median of x, y as the middle point.
    '''
    ind = np.array(np.where(pred > 0.5))
    [x_median, y_median] = np.median(ind, axis=1)
    x_median *= 256 / roi_size
    y_median *= 256 / roi_size
    x_min = int(max(0, x_median - win_size / 2))
    y_min = int(max(0, y_median - win_size / 2))
    x_max = x_min + win_size
    y_max = y_min + win_size
    return [x_min, x_max, y_min, y_max]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')