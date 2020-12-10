import glob

import annotation_set
import cv2
import h5py
import tqdm
from pandas import np

HOME = annotation_set.HOME
IMAGE_SIZE = annotation_set.IMAGE_SIZE
TYPE = annotation_set.TYPE

# Initializing the h5 file
hf = h5py.File(HOME +
               '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0],
                                                                              IMAGE_SIZE[1]), 'a')

images_dataset = hf['images']
annotated_dataset = hf['annotated']
anns_dataset = hf['annotations']
num_image = hf['images'].shape[0]

# Sampling new images
sampled_frames = list(images_dataset)
count = 0
for image_file in tqdm.tqdm(glob.glob('/alberto/deepposekit-data/datasets/dog/images/rs_dog/*.png')):
    img = cv2.imread(image_file)
    img = cv2.resize(img, IMAGE_SIZE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sampled_frames.append(img)
    count += 1
img_channel = sampled_frames[0].shape[2]

sampled_frames = np.concatenate(sampled_frames)
sampled_frames = np.reshape(sampled_frames, (num_image+count, IMAGE_SIZE[1], IMAGE_SIZE[0], img_channel))

# Reshaping in order to add new images

del hf['images']
hf.create_dataset('images', data=sampled_frames, maxshape=(None, None, None, None))
hf['annotated'].resize((num_image + count, annotated_dataset.shape[1]))
hf['annotations'].resize((num_image + count, anns_dataset.shape[1], anns_dataset.shape[2]))


print()
