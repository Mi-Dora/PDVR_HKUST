import imageio
import numpy as np


def convert2gif(img_paths: list, o_dir):
    # img_paths = ["img/1.jpg", "img/2.jpg", "img/3.jpg", "img/4.jpg"
    #     , "img/5.jpg", "img/6.jpg"]
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave(o_dir, gif_images, fps=1)


if __name__ == '__main__':
    dir_path = '../database/pic/'
    gif_index = np.load('../output_data/gif_index.npy')
    img_paths = []
    for index in gif_index:
        img_paths.append(dir_path+str(index[0])+'_'+str(index[1])+'.jpg')

    convert2gif(img_paths, '../output_data/output.gif')