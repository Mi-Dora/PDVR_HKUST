import imageio
import numpy as np
from tqdm import tqdm


def convert2gif(img_paths: list, o_dir):
    gif_images = []
    for path in tqdm(img_paths):
        gif_images.append(imageio.imread(path))
    imageio.mimsave(o_dir, gif_images, fps=1)


if __name__ == '__main__':
    dir_path = '../database/pic/'
    gif_index = np.load('../output_data/gif_index.npy')
    img_paths = []
    for index in gif_index:
        img_paths.append(dir_path+str(index[0])+'_'+str(index[1])+'.jpg')

    convert2gif(img_paths, '../output_data/output.gif')