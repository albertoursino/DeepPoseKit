import annotation_set
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

# models = sorted(glob.glob(HOME + '/deepposekit-data/datasets/{}/log_densenet.h5'.format(TYPE)))
# model = load_model(HOME + '/deepposekit-data/datasets/{}/log_densenet.h5'.format(TYPE))
#
# randomly_sampled_frames = []
# count = 0
# for image_file in tqdm.tqdm(glob.glob(
#         'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto/deepposekit-data/datasets/dog/DAVIS/Annotations/Full-Resolution/dog/*.png')):
#     count += 1
#     img = cv2.imread(image_file)
#     img = cv2.resize(img, IMAGE_SIZE)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     randomly_sampled_frames.append(img)
# img_channel = randomly_sampled_frames[0].shape[2]
#
# randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
# randomly_sampled_frames = np.reshape(randomly_sampled_frames, (count, IMAGE_SIZE[1], IMAGE_SIZE[0], img_channel))
#
# predictions = model.predict(randomly_sampled_frames, verbose=1)
#
# np.save(HOME + '/deepposekit-data/datasets/{}/predictions.npy'.format(TYPE), predictions)

predictions = np.load(HOME + '/deepposekit-data/datasets/{}/predictions.npy'.format(TYPE))

x, y, confidence = np.split(predictions, 3, -1)

data_generator = DataGenerator(
    HOME + '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0], IMAGE_SIZE[1]))

image, k = data_generator[0]
keypoints = predictions[0]

plt.figure(figsize=(5, 5))
image = image[0] if image.shape[-1] is 3 else image[..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
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

plt.show()

plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50,
            cmap=plt.cm.hsv, zorder=3)

plt.show()
