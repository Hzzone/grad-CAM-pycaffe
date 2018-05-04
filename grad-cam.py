# -*- coding: UTF-8 -*-
import sys
import shutil
import os
sys.path.insert(0, "caffe/python")
import caffe
import numpy as np
import dicom
import cv2
from scipy.misc import bytescale
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.cm as cm

def process(source, IMAGE_SIZE=227):
    ds = dicom.read_file(source)
    pixel_array = ds.pixel_array
    height, width = pixel_array.shape
    if height < width:
        pixel_array = pixel_array[:, int((width - height) / 2):int((width + height) / 2)]
    else:
        pixel_array = pixel_array[int((height - width) / 2):int((width + height) / 2), :]
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    # im = im / 256
    im = np.dstack((im, im, im))
    im = im[:, :, [2, 1, 0]]
    input_im = im.transpose((2, 0, 1))
    return im, input_im


caffe.set_mode_cpu()

net = caffe.Net("bone/alexnet_deploy.prototxt", "bone/all_alexnet_train_iter_8000.caffemodel", caffe.TEST)
net.blobs['data'].reshape(1, 3, 227, 227)

## output layer
final_layer = 'my-fc8'
### the last conv layer or any else you want to visualize
layer_name = 'conv1'



def visualize(input_im):
    net.blobs['data'].data[...] = input_im
    output = net.forward()
    predict_age = output['my-fc8'][0][0]

    label = np.zeros(net.blobs[final_layer].shape)
    label[0, 0] = predict_age
    imdiff = net.backward(diffs=['data', layer_name], **{net.outputs[0]: label})
    gradients = imdiff[layer_name]
    vis_grad = np.squeeze(gradients)

    mean_grads = np.mean(vis_grad, axis=(1, 2))

    activations = net.blobs[layer_name].data

    activations = np.squeeze(activations)

    n_nodes = activations.shape[0] # number of nodels
    vis_size = activations.shape[1:] #visualization shape

    vis = np.zeros(vis_size, dtype=np.float32)


    #generating saliency image
    for i in xrange(n_nodes):
        activation = activations[i, :, :]
        weight = mean_grads[i]
        weighted_activation = activation*weight
        vis += weighted_activation

    # We select only those activation which has positively contributed in prediction of given class

    vis = np.maximum(vis, 0)   # relu
    vis_img = Image.fromarray(vis, None)
    vis_img = vis_img.resize((227,227),Image.BICUBIC)
    vis_img = vis_img / np.max(vis_img)
    vis_img = Image.fromarray(np.uint8(cm.jet(vis_img) * 255))
    vis_img = vis_img.convert('RGB') # dropping alpha channel

    return vis_img


### for one image
# im, input_im = process('/Users/hzzone/Downloads/data/male/11.00-11.99/15696275')
# vis_img = visualize(input_im)
# im = Image.fromarray(im)
#
# heat_map = Image.blend(im, vis_img, 0.3)
# heat_map = np.array(heat_map)
#
# plt.imsave('h1.jpg', heat_map)
# plt.imshow(heat_map)
# plt.axis('off')
# plt.show()

### For a folder

save_dir = './data'
data_dir = u'/Volumes/Seagate Backup Plus Drive/深度学习数据集/盆骨'

for root, dirs, files in os.walk(data_dir):
    for file_name in files:
        dicom_file = os.path.join(root, file_name)
        im, input_im = process(dicom_file)
        vis_img = visualize(input_im)
        im = Image.fromarray(im)

        heat_map = Image.blend(im, vis_img, 0.3)
        heat_map = np.array(heat_map)

        save_path = os.path.join(save_dir, root.strip(data_dir))

        if not os.path.exists(save_path):
            os.makedirs(save_path)


        save_path = os.path.join(save_path, file_name)

        print(save_path)

        plt.imsave(save_path, heat_map)
