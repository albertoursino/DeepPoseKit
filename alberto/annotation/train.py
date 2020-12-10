from alberto.annotation import annotation_set
from pandas import np

from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia

from deepposekit.models import StackedHourglass
from deepposekit.models import load_model
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from deepposekit.callbacks import Logger, ModelCheckpoint

import time
from os.path import expanduser

HOME = annotation_set.HOME
IMAGE_SIZE = annotation_set.IMAGE_SIZE
TYPE = annotation_set.TYPE

data_generator = DataGenerator(
    datapath=HOME + '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0], IMAGE_SIZE[1]))

image, keypoints = data_generator[0]

plt.figure(figsize=(5, 5))
image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
for idx, jdx in enumerate(data_generator.graph):
    if jdx > -1:
        x1 = keypoints[0, idx, 0]
        x2 = keypoints[0, jdx, 0]
        if (0 <= x1 <= IMAGE_SIZE[0]) and (0 <= x2 <= IMAGE_SIZE[0]):
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )

# plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)

plt.show()

# Augmentation

augmenter = []

augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right

sometimes = []
sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                            shear=(-8, 8),
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode=ia.ALL)
                 )
sometimes.append(iaa.Affine(scale=(0.8, 1.2),
                            mode=ia.ALL,
                            order=ia.ALL,
                            cval=ia.ALL)
                 )
augmenter.append(iaa.Sometimes(0.75, sometimes))
augmenter.append(iaa.Affine(rotate=(-180, 180),
                            mode=ia.ALL,
                            order=ia.ALL,
                            cval=ia.ALL)
                 )
augmenter = iaa.Sequential(augmenter)

# image, keypoints = data_generator[0]
# image, keypoints = augmenter(images=image, keypoints=keypoints)
# plt.figure(figsize=(5, 5))
# image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
# cmap = None if image.shape[-1] is 3 else 'gray'
# plt.imshow(image, cmap=cmap, interpolation='none')
# for idx, jdx in enumerate(data_generator.graph):
#     if jdx > -1:
#         x1 = keypoints[0, idx, 0]
#         x2 = keypoints[0, jdx, 0]
#         if (0 <= x1 <= IMAGE_SIZE[0]) and (0 <= x2 <= IMAGE_SIZE[0]):
#             plt.plot(
#                 [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
#                 [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
#                 'r-'
#             )

plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50,
            cmap=plt.cm.hsv, zorder=3)

# plt.show()

train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=3,
                                    augmenter=augmenter,
                                    sigma=5,
                                    validation_split=0,
                                    use_graph=False,
                                    random_seed=1,
                                    graph_scale=1)
train_generator.get_config()

# n_keypoints = data_generator.keypoints_shape[0]
# batch = train_generator(batch_size=1, validation=False)[0]
# inputs = batch[0]
# outputs = batch[1]

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
# ax1.set_title('image')
# ax1.imshow(inputs[0, ..., 0], vmin=0, vmax=255)
#
# ax2.set_title('posture graph')
# ax2.imshow(outputs[0, ..., n_keypoints:-1].max(-1))
#
# ax3.set_title('keypoints confidence')
# ax3.imshow(outputs[0, ..., :n_keypoints].max(-1))
#
# ax4.set_title('posture graph and keypoints confidence')
# ax4.imshow(outputs[0, ..., -1], vmin=0)
# plt.show()

train_generator.on_epoch_end()

# Define a model

model = StackedHourglass(train_generator)

model.get_config()

# data_size = (10,) + data_generator.image_shape
# x = np.random.randint(0, 255, data_size, dtype="uint8")
# y = model.predict(x[:100], batch_size=100) # make sure the model is in GPU memory
# t0 = time.time()
# y = model.predict(x, batch_size=100, verbose=1)
# t1 = time.time()
# print(x.shape[0] / (t1 - t0))

# logger = Logger(validation_batch_size=10,
#                 # filepath saves the logger data to a .h5 file
#                 filepath=HOME + "/deepposekit-data/datasets/{}/log_densenet.h5".format(TYPE)
#                 )

# Remember, if you set validation_split=0 for your TrainingGenerator,
# which will just use the training set for model fitting,
# make sure to set monitor="loss" instead of monitor="val_loss".
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, verbose=1, patience=20)

model_checkpoint = ModelCheckpoint(
    HOME + "/deepposekit-data/datasets/{}/best_model_densenet.h5".format(TYPE),
    monitor="loss",
    # monitor="loss" # use if validation_split=0
    verbose=1,
    save_best_only=True,
)

early_stop = EarlyStopping(
    monitor="loss",
    # monitor="loss" # use if validation_split=0
    min_delta=0.001,
    patience=100,
    verbose=1
)

callbacks = [early_stop, reduce_lr, model_checkpoint]

model.fit(
    batch_size=5,
    validation_batch_size=10,
    callbacks=callbacks,
    # epochs=1000, # Increase the number of epochs to train the model longer
    epochs=20,
    n_workers=8,
    steps_per_epoch=None,
)

# model = load_model(
#     HOME + "/deepposekit-data/datasets/{}/best_model_densenet.h5".format(TYPE),
#     augmenter=augmenter,
#     generator=data_generator,
#     )
#
# model.fit(
#     batch_size=2,
#     validation_batch_size=10,
#     callbacks=callbacks,
#     epochs=50,
#     n_workers=8,
#     steps_per_epoch=None,
# )
