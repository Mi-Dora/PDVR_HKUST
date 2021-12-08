from matplotlib import pyplot as plt
import numpy as np
import random

x = 100 * 4

y1 = [random.randint(0, 22)/100 for i in range(1, 100)]
y2 = [random.randint(0, 24)/100 for i in range(1, 100)]
y3 = [random.randint(0, 19)/100 for i in range(1, 100)]
y4 = [random.randint(0, 23)/100 for i in range(0, 100)]
y1.append(random.randint(80, 100)/100)
y2.append(random.randint(80, 100)/100)
y3.append(random.randint(80, 100)/100)

y = y1 + y2 + y3 + y4

print(y)
plt.figure(figsize=(40, 2), dpi=300)
plt.plot(range(x), y, color="#8B4513")
plt.show()