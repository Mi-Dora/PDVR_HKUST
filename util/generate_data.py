import random
from generate_gif import pre_process, convert2gif
from save_video import save_video

import os


def select_frames(video_len):
    # make sure duration > 2s
    max_end = video_len - 60
    start = random.randint(0, max_end)
    min_start = start + 60
    end = random.randint(min_start, video_len)
    while end - start > 120:
        end = random.randint(min_start, video_len)

    return list(range(start, end + 1))


def select_frames_path(data_path, video_id, video_frames_list):
    img_paths = []
    for index in video_frames_list:
        img_paths.append(data_path + str(video_id) + '_' + str(index) + '.jpg')

    return img_paths


if __name__ == '__main__':

    original_data_path = '/Users/yingxiayin/ClassData/MachineLearning/dataset/'
    output_data_path = '/Users/yingxiayin/ClassData/MachineLearning/'

    # total num of video
    num = 5
    # a video contains x parts
    part_range = [5, 9]
    # original video data
    num_original_video = 13
    video_info_list = [[1, 1371], [2, 523], [3, 1380], [4, 1356], [5, 1371], [6, 1350], [7, 1350], [8, 1350], [9, 1350],
                       [10, 1350], [11, 1350], [12, 957], [13, 1276]]
    new_video_info = {}
    for i in range(num):
        print('generate gif image no.{:d}'.format(i))
        num_parts = random.randint(part_range[0], part_range[1])
        print('it has {:d} parts'.format(num_parts))
        single_video_info = [[], []]
        single_video_img = []
        for j in range(num_parts+1):
            video_list_id = random.randint(0, num_original_video-1)

            print(video_list_id)

            frame_list = select_frames(video_info_list[video_list_id][1])
            img_path = select_frames_path(original_data_path, video_info_list[video_list_id][0], frame_list)

            single_video_info[0] += [video_list_id+1 for x in range(len(frame_list))]
            single_video_info[1] += frame_list

            single_video_img += img_path

        new_video_info[str(i)] = single_video_info

        # gif
        new_img_paths = pre_process(single_video_img)
        # convert2gif(new_img_paths, output_data_path+str(i)+'.gif')
        save_video(new_img_paths, output_data_path+str(i)+'.gif', fps=25)

        print('Successfully')


