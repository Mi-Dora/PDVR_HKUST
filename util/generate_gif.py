import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image
import os


def pre_process(image_list):
    """处理一下图片大小"""
    for image_name in image_list:
        im = Image.open(image_name)
        im = im.resize((500, 500))		# 都搞成(300,300)尺寸的
        new_image_name = image_name.split('.jpg')[0]+'_new'+'.jpg'
        im.save(new_image_name)		# False指的是覆盖掉之前尺寸不规范的图片


def convert2gif(img_paths: list, o_dir):
    pre_process(img_paths)
    frames = []
    for path in tqdm(img_paths):
        frames.append(imageio.imread(path))
    imageio.mimsave(o_dir, frames, fps=1)

    for path in tqdm(img_paths):
        os.remove(path)


if __name__ == '__main__':
    dir_path = '../database/pic/'
    gif_index = np.load('../output_data/gif_index.npy')
    img_paths = []
    new_img_paths = []
    for index in gif_index:
        img_paths.append(dir_path+str(index[0])+'_'+str(index[1])+'.jpg')
        new_img_paths.append(dir_path+str(index[0])+'_'+str(index[1])+'_new.jpg')

    pre_process(img_paths)

    convert2gif(new_img_paths, '../output_data/output.gif')