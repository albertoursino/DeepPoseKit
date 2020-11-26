from deepposekit import Annotator

HOME = 'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/DeepPoseKit/alberto'

app = Annotator(
    datapath=HOME + '/deepposekit-data/datasets/dog/annotation_set.h5',
    dataset='images',
    skeleton=HOME + '/deepposekit-data/datasets/dog/skeleton.csv',
    shuffle_colors=False,
    text_scale=0.7)

app.run()
