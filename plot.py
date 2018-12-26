import numpy as np
import types
from matplotlib import *
import matplotlib.pyplot as plt
import convert_helper
import math
from interrupt_drop_model import *




come_p = 0.2
come_s = 0.2
serve_p = 0.1
serve_s = 0.03
nc = 15
k = 4

transfer_matrix = get_transfer_matrix(come_p, come_s, serve_p, serve_s, k, nc)
block_ratio = get_block_ratio(come_p, come_s, serve_p, serve_s, k, nc)
interrupt_ratio = get_interrupt_ratio(come_p, come_s, serve_p, serve_s, k, nc)
throughput_capcity = get_throughput_capcity(come_p, come_s, serve_p, serve_s, k, nc)
avg_length = get_avg_length(come_p, come_s, serve_p, serve_s, k, nc)
avg_delay = get_avg_delay(come_p, come_s, serve_p, serve_s, k, nc)

print("block ratio: %f" % block_ratio)
print("interrupt ratio : %f" % interrupt_ratio)
print("throughput_capcity: %f" % throughput_capcity)
print("average length: %f" % avg_length)
print("average delay: %f" % avg_delay)



x_axis = np.arange(1, 11, 1)
y_axis_1 = np.arange(x_axis.shape[0], dtype=float)
y_axis_2 = np.arange(x_axis.shape[0], dtype=float)
y_axis_3 = np.arange(x_axis.shape[0], dtype=float)
y_axis = np.arange(x_axis.shape[0], dtype=float)

for i in range(0,x_axis.shape[0]):
    block = get_block_ratio(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    interrupt = get_interrupt_ratio(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    throughput = get_throughput_capcity(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    total = block + interrupt + throughput
    y_axis[i] = block
    y_axis_1[i] = interrupt
    y_axis_2[i] = throughput
    y_axis_3[i] = total

fig = plt.figure()
fig.set_size_inches(w=19.20, h=10.80)
axes_1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes_1.set_xlabel("cache")
axes_1.plot(x_axis, y_axis, label="block_raio", marker=".")
axes_1.plot(x_axis, y_axis_1, label="interrupt_ratio", marker=".")
axes_1.plot(x_axis, y_axis_2, label="io", marker=".")
axes_1.plot(x_axis, y_axis_3, label="total", marker=".")
axes_1.grid()
axes_1.legend()
fig.show()
fig.savefig(fname="./result/total.svg", format="svg")
