import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# change this to your caffe root dir
caffe_root = '/home/chenqi-didi/Documents/work/caffe'
import os
import sys
sys.path.insert(0, caffe_root + '/python')

import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2
caffe.set_device(0)
caffe.set_mode_gpu()
print('Check Caffe OK!')

# load label map file
labelmap_file = 'data/labelmap_kitti.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
a = text_format.Merge(str(file.read()), labelmap)


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

# load model deploy prototxt and caffemodel weights
model_def = 'models/VGGNet/KITTI/SSD_414x125/deploy.prototxt'
model_weights = 'models/VGGNet/KITTI/SSD_414x125/VGG_KITTI_SSD_414x125_iter_120000.caffemodel'

net = caffe.Net(model_def,     
                model_weights,
                caffe.TEST)  

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

image_resize_width = 414
image_resize_height = 125
net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)

detect_image = 'data/test2.jpg'
image = caffe.io.load_image(detect_image)

transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

detections = net.forward()['detection_out']

# Parse the outputs.
det_label = detections[0, 0, :, 1]
det_conf = detections[0, 0, :, 2]
det_xmin = detections[0, 0, :, 3]
det_ymin = detections[0, 0, :, 4]
det_xmax = detections[0, 0, :, 5]
det_ymax = detections[0, 0, :, 6]
print('det_label', det_label)
print('det_conf', det_conf)
print('det_xmin', det_xmin)
print('det_ymin', det_ymin)
print('det_xmax', det_xmax)
print('det_ymax', det_ymax)

# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]
print('top_label', top_labels)
print('top_conf', top_conf)
print('top_xmin', top_xmin)
print('top_ymin', top_ymin)
print('top_xmax', top_xmax)
print('top_ymax', top_ymax)

image_mat = cv2.imread(detect_image, cv2.IMREAD_COLOR)
image_h = image_mat.shape[0]
image_w = image_mat.shape[1]

for i in range(0, top_conf.shape[0]):
    cv2.rectangle(image_mat, (int(round(top_xmin[i]*image_w)), int(round(top_ymin[i]*image_h))),
                  (int(round(top_xmax[i]*image_w)), int(round(top_ymax[i]*image_h))), (155, 25, 25), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_mat, top_labels[i], (int(round(top_xmin[i]*image_w)), int(round(top_ymin[i]*image_h))),
                font, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('image', image_mat)
cv2.waitKey(0)
cv2.destroyAllWindows()


