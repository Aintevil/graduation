import numpy as np
import convert_helper
import math

'''
    come_p      the arrival probability of PU
    come_s:     the arrival probability of SU
    serve_p:    the serve probability of PU
    serve_s:    the serve probability of SU in single channel
    k:          the amount of SUs in the system
    nc:         the amount of channels
'''

# function to generate static possibility matrix
def get_static_matrix(come_p, come_s, serve_p, serve_s, k, nc):
    if serve_s * nc >= 1:
        print("nc * serve_s = %2.2f, too big\n" % (serve_s*nc))
        exit(1)
    transfer_matrix = get_transfer_matrix(come_p, come_s, serve_p, serve_s, k, nc)
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
def get_transfer_matrix(come_p, come_s, serve_p, serve_s, k, nc):
    if k<0:
        print("invalid k")
        exit(-1)
    if k == 0:
        transfer_matrix = np.array(
            [
                [(1-come_s)*(1-come_p), (1-come_p)*come_s, come_p],
                [(1-come_p)*(1-come_s)*serve_s, (1-come_p)*((1-serve_s)+come_s*serve_s), come_p],
                [(1-come_p)*(1-come_s)*serve_p, (1-come_p)*come_s*serve_p, (1-serve_p) + serve_p*come_p]
            ]
        )
        return transfer_matrix

    transfer_matrix = np.empty((k+2, k+2),dtype=np.ndarray)
    transfer_matrix[0, 0] = np.array([(1-come_p)*(1-come_s)])
    transfer_matrix[0, 1] = np.array([(1-come_p)*come_s, come_p*(1-come_s)])
    transfer_matrix[0, 2] = np.array([0, (come_p*come_s)])
    serve_s_1 = math.ceil(1.0*nc/(k+1))*serve_s
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
        serve_s_m = math.ceil(m * nc / (k + 1.0)) * serve_s
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
def get_block_ratio(come_p, come_s, serve_p, serve_s, k, nc):
    static_matrix = get_static_matrix(come_p, come_s, serve_p, serve_s, k, nc)
    block_ratio = come_s*(static_matrix[0, 2*k+2]*((1-serve_p)+come_p*serve_p) +
                          static_matrix[0, 2*k+1]*((1-serve_s*nc)+come_p*serve_s*nc)
                          )
    return block_ratio


# function to get interrupt ratio
def get_interrupt_ratio(come_p, come_s, serve_p, serve_s, k, nc):
    static_matrix = get_static_matrix(come_p, come_s, serve_p, serve_s, k, nc)
    interrupt_ratio = 0.0
    for i in range(1, 2*k+2, 2):
        interrupt_ratio = interrupt_ratio + static_matrix[0, i]*come_p*(1-math.ceil((i//2+1)*nc/(k+1.0))*serve_s)
    return interrupt_ratio


# function to get throughput capcity
def get_throughput_capcity(come_p, come_s, serve_p, serve_s, k, nc):
    return come_s - get_block_ratio(come_p, come_s, serve_p, serve_s, k, nc) -\
           get_interrupt_ratio(come_p, come_s, serve_p, serve_s, k, nc)


# function to get average queue length
def get_avg_length(come_p, come_s, serve_p, serve_s, k, nc):
    static_matrix = get_static_matrix(come_p, come_s, serve_p, serve_s, k, nc)
    avg_length = 0.0
    for i in range(1,k+2,1):
        avg_length = avg_length + i*static_matrix[0, 2*i-1] + (i-1)*static_matrix[0, 2*i]
    return avg_length


# function to get average delay
def get_avg_delay(come_p, come_s, serve_p, serve_s, k, nc):
    avg_length = get_avg_length(come_p, come_s, serve_p, serve_s, k, nc)
    block_ratio = get_block_ratio(come_p, come_s, serve_p, serve_s, k, nc)
    return avg_length / (come_s - block_ratio)
