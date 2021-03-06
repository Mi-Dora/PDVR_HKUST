"""
Pipeline for PDVR
"""

from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing

import numpy as np
import torch
import tqdm
import json

from torch.utils.data import DataLoader

from Lstm import Lstm
from segmentation_DNN_model import MyDataset, SegmentationModel, RunDataSet
from utils import *
from scipy.spatial.distance import cdist

from extract_feature.model_tf import CNN_tf
from model import DNN
from feature_extraction import feature_extraction_images


def get_feature(img_list, save_path, model_path, cores=8, batch_sz=32):
    print('Current pid = ', os.getpid())
    vgg = CNN_tf('vgg', model_path)
    feature_extraction_images(vgg, cores, batch_sz,
                              img_list, save_path)


def get_change(query_embeddings):
    model = SegmentationModel(1000, 512, 128, 2)
    model.load_state_dict(torch.load('./output_data/judgement_model'))
    x_data = None
    y_data = []
    y_pred = None
    for i in range(0, query_embeddings.shape[0] - 1):
        new_data = np.vstack((query_embeddings[i][np.newaxis, :], query_embeddings[i + 1][np.newaxis, :]))[
            np.newaxis, ...]
        if x_data is None:
            x_data = new_data
        else:
            x_data = np.vstack((x_data, new_data))
    my_dataset = RunDataSet(x_data)
    # 实例化
    run_loader = DataLoader(dataset=my_dataset,  # 要传递的数据集
                            batch_size=32,  # 一个小批量数据的大小是多少
                            shuffle=False,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                            num_workers=0)
    for i, data in enumerate(run_loader):
        inputs = data
        result = model(inputs)
        if y_pred is None:
            y_pred = result.detach().numpy()
        else:
            y_pred = np.vstack((y_pred, result.detach().numpy()))
    return y_pred


def calculate_similarities(query_feature, features):
    pbar = tqdm.tqdm(total=len(query_feature))
    similarities_of_all = {}
    gif_index = []
    # for one frame(picture) of the query
    for i, query_pic_feature in enumerate(query_feature):

        # record every moment maximum dist, normalize it
        max_dist = 0
        min_dist_video = 10000000
        min_video_index = -1
        min_video_frame_index = -1
        for j, videos_features in enumerate(features):
            # find the min_dist value in each video
            min_dist_pic = 100000000
            min_pic_index = -1
            for k, pic_features in enumerate(videos_features):
                dist = np.sum((np.array(query_pic_feature) - np.array(pic_features)) ** 2)
                if dist < min_dist_pic:
                    min_dist_pic = dist
                    min_pic_index = k

            # add min_dist to (j+1)-th key's list
            if str(j + 1) in similarities_of_all:
                similarities_of_all[str(j + 1)].append(min_dist_pic)
            else:
                similarities_of_all[str(j + 1)] = [min_dist_pic]
            if min_dist_pic < min_dist_video:
                min_video_frame_index = min_pic_index
                min_video_index = j
                min_dist_video = min_dist_pic
            max_dist = max(min_dist_pic, max_dist)
        gif_index.append((min_video_index + 1, min_video_frame_index))
        # normalize every frame's distance (calculate final similarity)
        for values in similarities_of_all.values():
            values[i] = np.round(1 - values[i] / max_dist, decimals=6)
        pbar.update(1)
    pbar.close()
    np.save('./output_data/gif_index', np.array(gif_index))
    return similarities_of_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ql', '--query', type=str, default='fake_dataset/',
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    # parser.add_argument('-qp', '--query_path', type=str, default='query/pic',
    #                     help='Path to the .npy file that contains the global '
    #                          'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-m', '--model_path', type=str, default='model/',
                        help='Path to load the trained DML model')
    parser.add_argument('-db', '--database', type=str, default='database/',
                        help='Paths to the .npy files that contains the feature vectors '
                             'of the videos in the Database. Each line of the '
                             'file have to contain the video id (name of the video file) '
                             'and the full path to the corresponding .npy file, separated '
                             'by a tab character (\\t)')

    # parser.add_argument('-o', '--output_path', type=str, default='database/',
    #                     help='Output directory where the generated database files will be stored')
    parser.add_argument('-tf', '--tf_model', type=str, default='slim/vgg_16.ckpt',
                        help='Path to the .ckpt file of the pre-trained CNN model. '
                             'Required if Tensorflow framework is selected')

    args = vars(parser.parse_args())

    process_database_feature = True
    process_query_feature = True
    process_embedding = True

    if process_database_feature:
        print('Generating database...')
        db_lists = gen_database(img_path=os.path.join(args['database'], 'pic'), save_path=args['database'])
        print('Extract features from database')
        for i, db_list in enumerate(db_lists):
            # If do not use multiprocessing, there will be a memory overflow error
            p = multiprocessing.Process(target=get_feature,
                                        args=(os.path.join(args['database'] + db_list),
                                              args['database'], args['tf_model']))
            p.start()
            p.join()

    if process_query_feature:
        print('Processing query video...')
        p = multiprocessing.Process(target=get_feature,
                                    args=(args['query'] + 'q_list.txt', args['query'], args['tf_model']))
        p.start()
        p.join()

    if process_embedding:
        print('Embedding features...')
        query_features_file = os.path.join(args['query'], 'q_features.npy')
        query_features = np.load(query_features_file)

        model = DNN(query_features.shape[1],
                    args['model_path'],
                    load_model=True,
                    trainable=False)
        query_embeddings = model.embeddings(query_features)
        np.save(args['query'] + 'q_embedding.npy', query_embeddings)
        db_feature_files = []
        for root, _, files in os.walk(args['database']):
            for file in files:
                if file.endswith('_features.npy'):
                    db_feature_files.append(os.path.join(root, file))
        for file in db_feature_files:
            feature = np.load(file)
            db_embeddings = model.embeddings(feature)
            np.save(file[:-len('_features.npy')] + '_embedding.npy', db_embeddings)

    db_embedding_files = []
    for root, _, files in os.walk(args['database']):
        for file in files:
            if file.endswith('_embedding.npy'):
                vid = int(file.split('_')[0])
                db_embedding_files.append((os.path.join(root, file), vid))


    def getVid(x):
        return x[1]


    db_embedding_files.sort(key=getVid)
    db_embeddings = []
    for file in db_embedding_files:
        db_embeddings.append(np.load(file[0]))
    query_embeddings = np.load(args['query'] + 'q_embedding.npy')
    print('Computing similarity...')
    # change_label = get_change(query_embeddings)
    # lstm = Lstm()
    # change_label = np.load("output_data/y_pred.npy")
    # lstm.train_lstm(query_embeddings, db_embeddings, change_label)

    similarities = calculate_similarities(query_embeddings, db_embeddings)
    json_sim = json.dumps(similarities)
    with open('output_data/sim_output.json', 'w') as f:
        f.write(json_sim)
