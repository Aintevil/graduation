import numpy as np
import types
from matplotlib import *
import matplotlib.pyplot as plt
import convert_helper


# function to generate static possibility matrix
def get_static_matrix(come_p, come_s, serve_p, k):
    transfer_matrix = get_transfer_matrix(come_p, come_s, serve_p, k)
    last_matrix = np.zeros((1, 2*k+3), float)
    last_matrix[0, 0] = 1
    this_matrix = last_matrix.dot(transfer_matrix)
    diff = np.abs(this_matrix - last_matrix)
    while np.max(diff) > 0.0001:
        last_matrix = this_matrix
        this_matrix = last_matrix.dot(transfer_matrix)
        diff = np.abs(this_matrix - last_matrix)
    return this_matrix


# function to generate transfer possibility matrix
def get_transfer_matrix(come_p, come_s, serve_p, k):
    transfer_matrix = np.empty((k+2, k+2),dtype=np.ndarray)
    transfer_matrix[0, 0] = np.array([(1-come_p)*(1-come_s)])
    transfer_matrix[0, 1] = np.array([(1-come_p)*come_s, come_p*(1-come_s)])
    transfer_matrix[0, 2] = np.array([0, (come_p*come_s)])
    serve_s_1 = serve_p/(k+1)
    transfer_matrix[1, 0] = np.array([[(1-come_p)*(1-come_s)*serve_s_1], [(1-come_p)*(1-come_s)*serve_p]])
    transfer_matrix[1, 1] = np.array(
        [
            [(1-come_p)*((1-come_s)*(1-serve_s_1)+come_s*serve_s_1), come_p*(1-come_s)],
            [(1-come_p)*come_s*serve_p, (1-come_s)*(come_p*serve_p+(1-serve_p))]
        ]
    )
    transfer_matrix[1, 2] = np.array(
        [
            [(1-come_p)*come_s*(1-serve_s_1), come_p*come_s],
            [0, come_s*(come_p*serve_p + (1-serve_p))]
        ]
    )

    for m in range(2,k+2):
        serve_s_m = m*serve_p/(k+1)
        transfer_matrix[m, m-1] = np.array(
            [
                [(1-come_p)*(1-come_s)*serve_s_m, 0],
                [(1-come_p)*(1-come_s)*serve_p, 0]
            ]
        )
        if m != k+1:
            transfer_matrix[m, m] = np.array(
                [
                    [(1 - come_p) * ((1 - come_s) * (1 - serve_s_m) + come_s * serve_s_m), come_p * (1 - come_s)],
                    [(1 - come_p) * come_s * serve_p, (1 - come_s)*(come_p * serve_p + (1 - serve_p))]
                ]
            )
            transfer_matrix[m, m + 1] = np.array(
                [
                    [(1 - come_p) * come_s * (1 - serve_s_m), come_p * come_s],
                    [0, come_s * (come_p * serve_p + (1 - serve_p))]
                ]
            )
        else:
            transfer_matrix[m, m] = np.array(
                [
                    [(1-come_p)*(come_s+(1-come_s)*(1-serve_s_m)), come_p],
                    [(1-come_p)*come_s*serve_p, come_p*serve_p+(1-serve_p)]
                ]
            )

    # fill with zero
    for i in range(3,transfer_matrix.shape[1]):
        transfer_matrix[0,i] = np.array([0, 0])
    for i in range(2,transfer_matrix.shape[0]):
        transfer_matrix[i, 0] = np.array([
            [0],
            [0]
        ]
        )
    for i in range(1,transfer_matrix.shape[0]):
        for j in range(1,transfer_matrix.shape[1]):
            if j not in (i-1, i, i+1):
                transfer_matrix[i][j] = np.array([
                    [0, 0],
                    [0, 0]
                ]
                )

    # format matrix
    return convert_helper.cell2mat(transfer_matrix)


# function to get block ratio
def get_block_ratio(come_p, come_s, serve_p, k):
    static_matrix = get_static_matrix(come_p, come_s, serve_p, k)
    block_ratio = come_s*(static_matrix[0, 2*k+2]*((1-serve_p)+come_p*serve_p) +
                          static_matrix[0, 2*k+1]*((1-serve_p)+come_p*serve_p)
                          )
    return block_ratio


# function to get interrupt ratio
def get_interrupt_ratio(come_p, come_s, serve_p, k):
    static_matrix = get_static_matrix(come_p, come_s, serve_p, k)
    interrupt_ratio = 0.0
    for i in range(1, 2*k+2, 2):
        interrupt_ratio = interrupt_ratio + static_matrix[0, i] * come_p * (i//2 + 1) * serve_p /k
    return interrupt_ratio


# function to get throughput capcity
def get_throughput_capcity(come_p, come_s, serve_p, k):
    return come_s - get_block_ratio(come_p, come_s, serve_p, k) - get_interrupt_ratio(come_p, come_s, serve_p, k)


# function to get average queue length
def get_avg_length(come_p, come_s, serve_p, k):
    static_matrix = get_static_matrix(come_p, come_s, serve_p, k)
    avg_length = 0.0
    for i in range(1,k+2,1):
        avg_length = avg_length + i*static_matrix[0, 2*i-1] + (i-1)*static_matrix[0, 2*i]
    return avg_length


# function to get average delay
def get_avg_delay(come_p, come_s, serve_p, k):
    avg_length = get_avg_length(come_p, come_s, serve_p, k)
    throughput_capcity = get_throughput_capcity(come_p, come_s, serve_p, k)
    return avg_length/throughput_capcity



come_p = 0.1
come_s = 0.9
serve_p = 0.9
k = 100

block_ratio = get_block_ratio(come_p, come_s, serve_p, k)
interrupt_ratio = get_interrupt_ratio(come_p, come_s, serve_p, k)
throughput_capcity = get_throughput_capcity(come_p, come_s, serve_p, k)
avg_length = get_avg_length(come_p, come_s, serve_p, k)
avg_delay = get_avg_delay(come_p, come_s, serve_p, k)

print("block ratio: %s" % block_ratio)
print("interrupt ratio : %s" % interrupt_ratio)
print("throughput_capcity: %s" % throughput_capcity)
print("average length: %s" % avg_length)
print("average delay: %s" % avg_delay)

x_axis = np.arange(1, 100, 1)
y_axis = np.arange(x_axis.shape[0], dtype=float)

for i in range(0,x_axis.shape[0]):
    y_axis[i] = get_interrupt_ratio(come_p, come_s, serve_p, x_axis[i])

fig = plt.figure()
fig.set_size_inches(w=19.20, h=10.80)
axes_1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes_1.set_title("interrupt_ratio - cache k")
axes_1.set_xlabel("cache")
axes_1.set_ylabel("interrupt ratio")
axes_1.grid()
axes_1.plot(x_axis, y_axis, label="interrupt ratio")
axes_1.legend()
fig.show()
fig.savefig(fname="./result/interrupt_ratio.svg", format="svg")

for i in range(0,x_axis.shape[0]):
    y_axis[i] = get_throughput_capcity(come_p, come_s, serve_p, x_axis[i])
fig = plt.figure()
fig.set_size_inches(w=19.20, h=10.80)
axes_1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes_1.set_title("throughput_capcity - cache k")
axes_1.set_xlabel("cache")
axes_1.set_ylabel("throughput_capcity")
axes_1.grid()
axes_1.plot(x_axis, y_axis, label="throughput_capcity")
axes_1.legend()
fig.show()
fig.savefig(fname="./result/throughput_capcity.svg", format="svg")

for i in range(0,x_axis.shape[0]):
    y_axis[i] = get_avg_length(come_p, come_s, serve_p, x_axis[i])
fig = plt.figure()
fig.set_size_inches(w=19.20, h=10.80)
axes_1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes_1.set_title("average length - cache k")
axes_1.set_xlabel("cache")
axes_1.set_ylabel("average length")
axes_1.grid()
axes_1.plot(x_axis, y_axis, label="average length")
axes_1.legend()
fig.show()
fig.savefig(fname="./result/average_length.svg", format="svg")

for i in range(0,x_axis.shape[0]):
    y_axis[i] = get_avg_delay(come_p, come_s, serve_p, x_axis[i])
fig = plt.figure()
fig.set_size_inches(w=19.20, h=10.80)
axes_1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes_1.set_title("average delay - cache k")
axes_1.set_xlabel("cache")
axes_1.set_ylabel("average delay")
axes_1.grid()
axes_1.plot(x_axis, y_axis, label="average delay")
axes_1.legend()
fig.show()
fig.savefig(fname="./result/average_delay.svg", format="svg")




