

# Sampling new images
import glob

import h5py
import tqdm
import cv2
from annotation_set import IMAGE_SIZE, HOME, TYPE
from pandas import np

hf = h5py.File(HOME +
               '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0],
                                                                              IMAGE_SIZE[1]), 'a')

sampled_frames = []
count = 0
for image_file in tqdm.tqdm(glob.glob('C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto/deepposekit-data/datasets/dog/images/dog_samples/*.png')):
    img = cv2.imread(image_file)
    resized_img = cv2.resize(img, IMAGE_SIZE)
    gray_res_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_res_img = np.reshape(gray_res_img, (IMAGE_SIZE[1], IMAGE_SIZE[0], 1))
    sampled_frames.append(gray_res_img)
    count += 1

for image_file in tqdm.tqdm(glob.glob('C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto/deepposekit-data/datasets/dog/images/dog_agility/*.png')):
    img = cv2.imread(image_file)
    resized_img = cv2.resize(img, IMAGE_SIZE)
    gray_res_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_res_img = np.reshape(gray_res_img, (IMAGE_SIZE[1], IMAGE_SIZE[0], 1))
    sampled_frames.append(gray_res_img)
    count += 1

for image_file in tqdm.tqdm(glob.glob('/alberto/deepposekit-data/datasets/dog/images/rs_dog/*.png')):
    img = cv2.imread(image_file)
    resized_img = cv2.resize(img, IMAGE_SIZE)
    gray_res_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_res_img = np.reshape(gray_res_img, (IMAGE_SIZE[1], IMAGE_SIZE[0], 1))
    sampled_frames.append(gray_res_img)
    count += 1

sampled_frames = np.concatenate(sampled_frames)
sampled_frames = np.reshape(sampled_frames, (count, IMAGE_SIZE[1], IMAGE_SIZE[0], 1))

del hf['images']
hf.create_dataset('images', data=sampled_frames, maxshape=(None, None, None, None))

