from matplotlib import pyplot as plt
import json
import numpy as np
from scipy.interpolate import make_interp_spline

''' 
draw the picture from dict, where dict is form of
d: {'id':[1,2,3,4],...}, id is the query video id and list is the similarity in every frame
interval: the interval second between frames (adjacent frames) 
'''


def draw(d: dict, interval: int):
    colors = ['#BBFFFF', '#90EE90', '#6A5ACD', '#8B8682', '#FFB6C1', '#B0C4DE', '#EE6363', '#CDC9A5',
              '#FFD700', '#8B4513', '#B03060', '#CDC9A5', '#FFFFF0']

    # save max similarity value each moment
    xs = np.linspace(0, len(d['1']), 10000)
    max_sim = [0] * len(d['1'])
    # x coordinate
    for key in d.keys():
        # save maximum similarity
        for i, sim in enumerate(d[key]):
            max_sim[i] = max(max_sim[i], sim)
        x = list(range(0, len(d[key]), interval))
        model = make_interp_spline(x, d[key])
        # plt.plot(x, d[key], color=colors[int(key)-1], marker=None, label='video'+str(key), linestyle='-')
        ys = model(xs)
        # draw each line
        plt.plot(xs, ys, color=colors[int(key)-1], marker=None, label='video'+str(key), linestyle='-')
    plt.title('Similarities between query video and Database')
    plt.xlabel('Time /s')
    plt.ylabel('Similarity')
    plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left')
    plt.show()


def draw_another(d: dict, interval: int):
    colors = ['#BBFFFF', '#90EE90', '#6A5ACD', '#8B8682', '#FFB6C1', '#B0C4DE', '#EE6363', '#CDC9A5',
              '#FFD700', '#8B4513', '#B03060', '#CDC9A5', '#FFFFF0']

    # save max similarity value each moment
    xs = np.linspace(0, len(d['1']), 10000)
    max_sim = [0] * len(d['1'])
    # for key in d.keys():
    #     # save maximum similarity
    #     for i, sim in enumerate(d[key]):
    #         if sim < 0.3:
    #             d[key][i] = 0

    # x coordinate
    for key in d.keys():
        # save maximum similarity
        for i, sim in enumerate(d[key]):
            max_sim[i] = max(max_sim[i], sim)
        x = list(range(0, len(d[key]), interval))
        # model = make_interp_spline(x, d[key])
        # ys = model(xs)
        # # draw each line
        if key == '1' or key == '3' or key == '4' or key == '5' or key == '10':
            plt.plot(x, d[key], color=colors[int(key) - 1], marker=None, label='video' + str(key), linestyle='-')

            # plt.plot(xs, ys, color=colors[int(key)-1], marker=None, label='video'+str(key), linestyle='-')
    plt.title('Similarities between query video and Database')
    plt.xlabel('Time /s')
    plt.ylabel('Similarity')
    plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left')
    plt.show()


if __name__ == '__main__':
    with open('../output_data/sim_output.json', 'r') as f:
        content = f.read()
    sim_dict = json.loads(content)
    draw(sim_dict, 1)

    with open('../output_data/sim_output.json', 'r') as f:
        content = f.read()
    sim_dict = json.loads(content)
    draw_another(sim_dict, 1)