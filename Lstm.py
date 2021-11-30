import numpy as np


class Lstm():

    def __init__(self):
        self.w = []

    def train_lstm(self, query_feature, features, frame_change):
        # h = np.zeros((len(query_feature),2))
        # s = np.zeros((128,128))
        gif_index = []
        similarities_of_all = {}
        weight = 0.01

        for i, query_frame in enumerate(query_feature):

            min_value_total = 100000.0
            min_video = 0
            min_position_total = 0
            min_value = 0.0
            min_position = -1
            for j, videos_features in enumerate(features):
                feature_count = len(videos_features)
                query_frame_matrix = np.repeat([query_frame], [feature_count], axis=0)
                dist = np.sum((videos_features - query_frame_matrix) ** 2, axis=1)
                if h == i and s == j:
                    dist = np - weight
                min_value = np.min(dist)
                min_position = np.where(dist == np.min(dist))
                if min_value_total > min_value:
                    min_value_total = min_value
                    min_position_total = min_position
                    min_video = j
            if (frame_change[i][0] > frame_change[i][1]):
                h = min_video
                s = min_video
            else:
                h = 0
                s = 0

            print(min_value_total)

            gif_index.append((min_video + 1, min_position_total[0][0]))

        np.save('./output_data/gif_index', np.array(gif_index))
