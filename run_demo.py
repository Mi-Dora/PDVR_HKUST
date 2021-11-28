"""
Pipeline for PDVR
"""

from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing

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
    parser.add_argument('-ql', '--query_list', type=str, default='query/q_list.txt',
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-qp', '--query_path', type=str, default='query/',
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-m', '--model_path', type=str, default='model/',
                        help='Path to load the trained DML model')
    parser.add_argument('-db', '--database', type=str, default='database/',
                        help='Paths to the .npy files that contains the feature vectors '
                             'of the videos in the Database. Each line of the '
                             'file have to contain the video id (name of the video file) '
                             'and the full path to the corresponding .npy file, separated '
                             'by a tab character (\\t)')

    parser.add_argument('-o', '--output_path', type=str, default='database/',
                        help='Output directory where the generated database files will be stored')
    parser.add_argument('-tf', '--tf_model', type=str, default='slim/vgg_16.ckpt',
                        help='Path to the .ckpt file of the pre-trained CNN model. '
                             'Required if Tensorflow framework is selected')

    args = vars(parser.parse_args())

    process_database = False
    process_query = False

    if process_database:
        print('Generating database...')
        db_lists = gen_database(img_path=os.path.join(args['database'], 'pic'), save_path=args['database'])
        print('Extract features from database')
        for i, db_list in enumerate(db_lists):
            # If do not use multiprocessing, there will be a memory overflow error
            p = multiprocessing.Process(target=get_feature,
                                        args=(os.path.join(args['database'] + db_list),
                                              args['output_path'], args['tf_model']))
            p.start()
            p.join()

    if process_query:
        print('Processing query video...')
        p = multiprocessing.Process(target=get_feature, args=(args['query_list'], args['query_path'], args['tf_model']))
        p.start()
        p.join()

    print('Embedding features...')
    query_features_file = os.path.join(args['query_path'], 'q_features.npy')
    query_features = np.load(query_features_file)

    model = DNN(query_features.shape[1],
                args['model_path'],
                load_model=True,
                trainable=False)
    query_embeddings = model.embeddings(query_features)
    np.save(args['query_path'] + 'q_embedding.npy', query_embeddings)
    feature_files = []
    for root, _, files in os.walk(args['output_path']):
        for file in files:
            if file.endswith('_features.npy'):
                feature_files.append(os.path.join(root, file))
    for file in feature_files:
        feature = np.load(file)
        db_embeddings = model.embeddings(feature)
        np.save(file[:-len('_features.npy')]+'_embedding.npy', db_embeddings)


    # similarities = calculate_similarities(cc_dataset['queries'], query_embeddings)

