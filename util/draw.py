from matplotlib import pyplot as plt

''' 
draw the picture from dict, where dict is form of
d: {'id':[1,2,3,4],...}, id is the query video id and list is the similarity in every frame
interval: the interval second between frames (adjacent frames) 
'''


def draw(d: dict, interval: int):
    colors = ['#BBFFFF', '#008B8B', '#76EEC6', '#00CD66', '#7B68EE', '#B0C4DE', '#EE6363', '#556B2F',
              '#FFD700', '#8B4513', '#B03060', '#CDC9A5', '#FFFFF0']
    color_index = 0

    # save max similarity value each moment
    max_sim = [0] * len(d['0'])
    # x coordinate
    x = list(range(0, len(d['0']) * interval, interval))
    for key in d.keys():
        # save maximum similarity
        for i, sim in enumerate(d[key]):
            max_sim[i] = max(max_sim[i], sim)
        # draw each line
        plt.plot(x, d[key], markersize='3', color=colors[color_index], marker='o', label='video'+str(key), linestyle='-')
        color_index += 1

    # draw max similarity line
    plt.plot(x, max_sim, markersize='5', color='#000000', marker='o', label='max similarity', linestyle='-')
    color_index += 1
    plt.title('Similarities between query video and Database')
    plt.xlabel('Time /s')
    plt.ylabel('Similarity')
    plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left')
    plt.show()