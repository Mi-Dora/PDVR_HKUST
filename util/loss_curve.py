from matplotlib import pyplot as plt
import numpy as np


with open('../output_data/loss_log.txt', 'r') as loss_f:
    lines = loss_f.readlines()
loss = np.zeros(5000)
for i, line in enumerate(lines[:5000]):
    loss[i] = np.float(line)

plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('../output_data/loss.png')
plt.show()




