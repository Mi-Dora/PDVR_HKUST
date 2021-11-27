import numpy as np

video = np.load('1_1_Y.flv')
feature = np.load('output_data/cc_web_video_features.npy')
triplet = np.load('output_data/cc_web_video_triplets.npy')
a = triplet[10000:11000, :]
pass



