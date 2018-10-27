import pydicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import izip
from utils import center_crop, lr_poly_decay, get_SAX_SERIES

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = '../Data/'
TEST_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart1',
                            'OnlineDataContours')
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, 'challenge_online/challenge_online')
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart2',
                            'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, 'challenge_validation')

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'TrainingDataContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')
SIZE = 256

def shrink_case(case):
    toks = case.split('-')
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return '-'.join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
        self.ctr = np.loadtxt(self.ctr_path, delimiter=' ').astype('int')
    
    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)
    
    __repr__ = __str__


def read_contour(contour, data_path):
    filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(data_path, contour.case, filename)
    f = pydicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask


def map_all_contours(contour_path, shuffle=False):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-icontour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours


def export_all_contours(contours, data_path):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), SIZE, SIZE, 1))
    masks = np.zeros((len(contours), SIZE, SIZE, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        if img.shape[0] > SIZE:
            img = center_crop(img, SIZE)
            mask = center_crop(mask, SIZE)
        images[idx] = img
        masks[idx] = mask

    return images, masks

def prepareDataset(contour_path, img_path):
    contours = map_all_contours(contour_path)
    img, mask = export_all_contours(contours, img_path)
    return img, mask, contours

def reformDataXY(img, ROI, img_size = 64,  mask_size = 32):
    '''
        Reform the image data and ROI for model
        @param:
            img: the original image, shape (N, 256, 256, 1)
            ROI: the bounding box of region of interest, shape (N, mask_size, mask_size)
            img_size: size image used for the model, default 64
            mask_size: size of mask used for the model, default 32
        @return:
            X: the reformed data field, shape (N, img_size, img_size, 1)
            Y: the reformed ground truth, shape (N, 1, mask_size, mask_size)
    '''
    X = np.zeros((img.shape[0], img_size, img_size, 1))
    for i in range(X.shape[0]):
        X[i,:,:,0] = cv2.resize(img[i,:,:,0], (img_size, img_size), interpolation = cv2.INTER_LINEAR)
    Y = np.array(ROI).reshape((len(ROI),1, mask_size, mask_size))
    return X, Y



def get_ROI(contours, shape_out = 32, img_size = 256):
    '''
        Given the path to the mask, return ROI -- the bounding box with size shape_out
        @param
            countour_path: the path to the mask dir
            shape_out: the size of bounding box, default 32
            img_size: original size of image, default 256
        @return
            ROI: the bounding box computed based on ground truth
    '''
    ROI = []
    for i in range(len(contours)):
        c = contours[i].ctr
        X_min, Y_min = c[:,0].min(), c[:,1].min()
        X_max, Y_max = c[:,0].max(), c[:,1].max()  
        w = X_max - X_min
        h = Y_max - Y_min
        roi_single = np.zeros((img_size, img_size))
        if w > h :
            roi_single[int(Y_min - (w -h)/2):int(Y_max + (w -h)/2), int(X_min):int(X_max)] = 1.0
        else :
            roi_single[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
        ROI.append(cv2.resize(roi_single, (shape_out, shape_out), interpolation = cv2.INTER_NEAREST))
    return ROI



    