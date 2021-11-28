# Copyright 2018 Giorgos Kordopatis-Zilos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Implementation of the evaluation process based on CC_WEB_VIDEO dataset.
"""

from __future__ import division
from __future__ import print_function

import argparse
import os

from utils import *
from model import DNN
from tqdm import tqdm
from scipy.spatial.distance import cdist

from extract_feature.model_tf import CNN_tf


def extract_feature(image_list):
    vgg = CNN_tf(args['network'].lower(), args['tf_model'])



def calculate_similarities(video_query_f, database):

    dists = np.zeros(video_query_f.shape[0])
    high_sim_db_idxs = -np.ones(video_query_f.shape[0])
    dist_max = 0
    for i, img_query_f in enumerate(video_query_f):
        for db_id, v_db_f in enumerate(database):
            for img_db_f in v_db_f:
                dist = np.nan_to_num(cdist(img_query_f, img_db_f, metric='euclidean'))
                if dist < dist[i]:
                    dists[i] = dist
                    high_sim_db_idxs[i] = db_id
                if dist_max < dist:
                    dist_max = dist
    similarities = np.round(1 - dists / dist_max, decimals=6)
    return similarities, high_sim_db_idxs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-es', '--query_set', type=str, default='output_data/vgg_features.npy',
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-m', '--model_path', type=str, default='model/',
                        help='Path to load the trained DML model')
    parser.add_argument('-f', '--fusion', type=str, default='Early',
                        help='Processed dataset. Options: Early and Late. Default: Early')
    parser.add_argument('-db', '--database', type=str, default='database/',
                        help='Paths to the .npy files that contains the feature vectors '
                             'of the videos in the Database. Each line of the '
                             'file have to contain the video id (name of the video file) '
                             'and the full path to the corresponding .npy file, separated '
                             'by a tab character (\\t)')
    parser.add_argument('-pl', '--positive_labels', type=str, default='ESLMV',
                        help='Labels in CC_WEB_VIDEO datasets that '
                             'considered posetive. Default=\'ESLMV\'')

    parser.add_argument('-n', '--network', type=str, default='vgg',
                        help='Name of the network')
    parser.add_argument('-o', '--output_path', type=str, default='test/',
                        help='Output directory where the generated files will be stored')
    parser.add_argument('-v', '--video_list', type=str,
                        help='List of videos to extract features')
    parser.add_argument('-i', '--image_list', type=str, default='pic/image_list.txt',
                        help='List of images to extract features')
    parser.add_argument('-tf', '--tf_model', type=str, default='slim/vgg_16.ckpt',
                        help='Path to the .ckpt file of the pre-trained CNN model. '
                             'Required if Tensorflow framework is selected')
    parser.add_argument('-pt', '--prototxt', type=str,
                        help='Path to the .prototxt file of the pre-trained CNN model. '
                             'Required if Caffe framework is selected')
    parser.add_argument('-cm', '--caffemodel', type=str,
                        help='Path to the .caffemodel file of the pre-trained CNN model. '
                             'Required if Caffe framework is selected')
    parser.add_argument('-c', '--cores', type=int, default=10,
                        help='Number of CPU cores for the parallel load of images or video')
    parser.add_argument('-b', '--batch_sz', type=int, default=32,
                        help='Number of the images fed to the CNN network at once')



    args = vars(parser.parse_args())


    print('Loading data...')
    db_list = gen_database(img_path=os.path.join(args['database'], 'pic'), save_path=args['database'])
    query_features = load_features(args['query_set'])
    database = []
    # for root, _, files in os.walk(args['database']):
    #     for file in files:
    #         video = np.load(os.path.join(root, file))
    #         database.append(video)

    print('Loading model...')
    model = DNN(query_features.shape[1],
                args['model_path'],
                load_model=True,
                trainable=False)

    print('Computing embeddings')
    if args['fusion'].lower() == 'early':
        print('Fusion type: Early')
        print('Extract video embeddings...')
        query_embeddings = model.embeddings(query_features)
        db = []
        for db in database:
            pass

    # else:
    #     print('Fusion type: Late')
    #     print('Extract video embeddings...')
    #
    #     assert args['database'] is not None, \
    #         'Argument \'--database\' must be provided for Late fusion'
    #     feature_files = load_feature_files(args['database'])
    #
    #     query_embeddings = np.zeros((len(cc_dataset['index']), model.embedding_dim))
    #     for i, video_id in enumerate(tqdm(cc_dataset['index'])):
    #         if video_id in feature_files:
    #             features = load_features(feature_files[video_id])
    #             embeddings = model.embeddings(normalize(features))
    #             embeddings = embeddings.mean(0, keepdims=True)
    #             query_embeddings[i] = normalize(embeddings, zero_mean=False)




    # similarities = calculate_similarities(cc_dataset['queries'], query_embeddings)

