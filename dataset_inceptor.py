import numpy as np
import pickle

b = pickle.load('datasets/cc_web_video.pickle', 'r')

feature = np.load('output_data/1_vgg.npy')
triplet = np.load('output_data/cc_web_video_triplets.npy')
a = triplet[10000:11000, :]
pass



