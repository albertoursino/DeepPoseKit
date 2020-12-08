from alberto.annotation import annotation_set
from deepposekit import Annotator

HOME = annotation_set.HOME
IMAGE_SIZE = annotation_set.IMAGE_SIZE
TYPE = annotation_set.TYPE

app = Annotator(
    datapath=HOME + '/deepposekit-data/datasets/{}/annotation_set_{}_{}.h5'.format(TYPE, IMAGE_SIZE[0], IMAGE_SIZE[1]),
    dataset='images',
    skeleton=HOME + '/deepposekit-data/datasets/{}/dog_skeleton.csv'.format(TYPE),
    shuffle_colors=False,
    text_scale=0.3)

app.run()
