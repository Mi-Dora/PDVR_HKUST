import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image
import os


def pre_process(image_list):
    """处理一下图片大小"""
    print('precess images\' size:')
    new_img_list = []
    for image_name in tqdm(image_list):
        im = Image.open(image_name)
        im = im.resize((500, 300))		# 都搞成(N,N)尺寸的
        new_image_name = image_name.split('.jpg')[0]+'_new'+'.jpg'
        new_img_list.append(new_image_name)
        im.save(new_image_name)		# False指的是覆盖掉之前尺寸不规范的图片
    return new_img_list


def convert2gif(img_list: list, o_dir):
    print('save gif image:')
    frames = []
    for path in tqdm(img_list):
        frames.append(imageio.imread(path))
    imageio.mimsave(o_dir, frames, fps=1)

    print('remove all resized images:')
    for path in tqdm(img_list):
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    dir_path = '../database/pic/'
    gif_index = np.load('../output_data/gif_index.npy')
    img_paths = []
    for index in gif_index:
        img_paths.append(dir_path+str(index[0])+'_'+str(index[1])+'.jpg')

    new_img_paths = pre_process(img_paths)

    convert2gif(new_img_paths, '../output_data/output.gif')