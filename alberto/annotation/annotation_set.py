import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from deepposekit.io import DataGenerator, initialize_dataset

HOME = 'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto'
IMAGE_SIZE = (512, 256)
TYPE = 'dog'


# This class loads and samples images from a video, defines a keypoint skeleton, and saves the data to a file for labelling with keypoints.

def main():
    # reader = VideoReader(HOME + '/deepposekit-data/datasets/fly/video.avi', gray=True)
    # frame = reader[0]  # read a frame
    # reader.close()
    # print(frame.shape)

    # reader = VideoReader(
    #     HOME + '/deepposekit-data/datasets/fly/video.avi',
    #     batch_size=100, gray=True)
    # randomly_sampled_frames = []
    # for idx in tqdm.tqdm(range(len(reader) - 1)):
    #     batch = reader[idx]
    #     random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
    #     randomly_sampled_frames.append(random_sample)
    # reader.close()
    #
    # randomly_sampled_frames = np.concatenate(randomly_sampled_frames)

    randomly_sampled_frames = []
    count = 0
    for image_file in tqdm.tqdm(glob.glob(
            'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto/deepposekit-data/datasets/dog/dog_samples/*.png')):
        count += 1
        img = cv2.imread(image_file)
        img = cv2.resize(img, IMAGE_SIZE)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        randomly_sampled_frames.append(img)
    img_channel = randomly_sampled_frames[0].shape[2]

    randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
    randomly_sampled_frames = np.reshape(randomly_sampled_frames, (count, IMAGE_SIZE[1], IMAGE_SIZE[0], img_channel))
    var = randomly_sampled_frames.shape
    print(var)

    # kmeans = KMeansSampler(n_clusters=10, max_iter=1000, n_init=10, batch_size=100, verbose=True)
    # kmeans.fit(randomly_sampled_frames)
    # kmeans.plot_centers(n_rows=2)
    # plt.show()
    # kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(randomly_sampled_frames, n_samples_per_label=10)
    # var = kmeans_sampled_frames.shape
    # print(var)

    initialize_dataset(images=randomly_sampled_frames,
                       datapath=HOME + '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE,
                                                                                                      IMAGE_SIZE[0],
                                                                                                      IMAGE_SIZE[1]),
                       skeleton=HOME + '/deepposekit-data/datasets/{}/dog_skeleton.csv'.format(TYPE))

    data_generator = DataGenerator(
        HOME + '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0], IMAGE_SIZE[1]),
        mode="full")

    image, keypoints = data_generator[0]

    plt.figure(figsize=(5, 5))
    image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )
    plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50,
                cmap=plt.cm.hsv, zorder=3)
    plt.show()


if __name__ == '__main__':
    main()
