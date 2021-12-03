from matplotlib import pyplot as plt
import numpy as np

def load_loss(f_path):
    with open(f_path, 'r') as loss_f:
        lines = loss_f.readlines()
    loss = np.zeros(len(lines))
    for i, line in enumerate(lines):
        loss[i] = np.float(line)
    return loss


l3 = '../output_data/loss_log.txt'
l2 = '../output_data/loss_log2.txt'
l4 = '../output_data/loss_log4.txt'
loss2 = load_loss(l2)
loss3 = load_loss(l3)
loss4 = load_loss(l4)
plt.plot(loss2[:5000], label='2-fc layer')
plt.plot(loss3[:5000], label='3-fc layer')
plt.plot(loss4[:5000], label='4-fc layer')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('../output_data/loss.png')
plt.show()




