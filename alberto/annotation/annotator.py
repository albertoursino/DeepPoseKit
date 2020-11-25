from deepposekit import Annotator
from os.path import expanduser
import glob

HOME = 'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto'

app = Annotator(
    datapath=HOME + '/deepposekit-data/datasets/dog/example_annotation_set.h5',
    dataset='images',
    skeleton=HOME + '/deepposekit-data/datasets/dog/skeleton.csv',
    shuffle_colors=False,
    text_scale=1)

app.run()
