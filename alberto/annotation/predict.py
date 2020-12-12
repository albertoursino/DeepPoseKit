import annotation_set
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

from deepposekit.models import load_model
from deepposekit.io import DataGenerator, VideoReader, VideoWriter
from deepposekit.io.utils import merge_new_images

import tqdm
import time

from scipy.signal import find_peaks

from os.path import expanduser

HOME = annotation_set.HOME
IMAGE_SIZE = (512, 256)
TYPE = annotation_set.TYPE

# models = sorted(glob.glob(HOME + '/deepposekit-data/datasets/{}/model_densenet.h5'.format(TYPE)))
# model = load_model(HOME + '/deepposekit-data/datasets/{}/model_densenet.h5'.format(TYPE))
#
# hf = h5py.File(HOME +
#                '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0],
#                                                                               IMAGE_SIZE[1]), 'r')
#
# images = hf['images']
#
# predictions = model.predict(images, verbose=1)
#
# np.save(HOME + '/deepposekit-data/datasets/{}/predictions.npy'.format(TYPE), predictions)

predictions = np.load(HOME + '/deepposekit-data/datasets/{}/predictions.npy'.format(TYPE))

# x, y, confidence = np.split(predictions, 3, -1)

data_generator = DataGenerator(
    HOME + '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0], IMAGE_SIZE[1]))

for i in range(len(data_generator)):
    image, k = data_generator[i]
    keypoints = predictions[i]
    plt.figure(figsize=(5, 5))
    image = image[0] if image.shape[-1] is 3 else image[..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image[0], cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[idx, 0], keypoints[jdx, 0]],
                [keypoints[idx, 1], keypoints[jdx, 1]],
                'r-'
            )
    plt.scatter(keypoints[:, 0], keypoints[:, 1],
                c=np.arange(data_generator.keypoints_shape[0]),
                s=50, cmap=plt.cm.hsv, zorder=3)

    plt.savefig(HOME + '/predicted-images/image_{}'.format(i))

