#! /usr/bin/python2
# coding=utf-8

import numpy as np
import types
import matplotlib
import matplotlib.pyplot as plt
import convert_helper
import math
import interrupt_drop_model as drop_model
import interrupt_return_model as return_model

come_p = 0.08
come_s = 0.3
serve_p = 0.2
serve_s = 0.03
nc = 15


def compare_delay():
    # #####################compare avg_delay##############################
    x_axis = np.arange(1, 11, 1)
    delay_drop = np.arange(1, 11, 1, dtype="float")
    delay_return = np.arange(1, 11, 1, dtype="float")

    fig = plt.figure(figsize=(7.00, 5.25), clear=True)
    axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axe_1.tick_params(top='on', right='on', direction='in')
    come_p = 0.16
    for i in range(0, x_axis.shape[0]):
        delay_drop[i] = drop_model.get_avg_delay(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
        delay_return[i] = return_model.get_avg_delay(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    # axe_1.plot(x_axis, delay_drop, linewidth=0.5, color="black", marker=".",
    #            label=r"$\lambda_1=%s$ Model Drop" % serve_p)
    # axe_1.plot(x_axis, delay_return, linewidth=0.5, color="gray", marker=".",
    #            label=r"$\lambda_1=%s$ Model Return" % serve_p)
    axe_1.plot(x_axis, delay_drop, linewidth=0.5, color="black", marker=".", label=r"$\lambda_1=%s$" % come_p)
    axe_1.plot(x_axis, delay_return, linewidth=0.5, color="gray", marker=".")
    axe_1.annotate("Model Drop", xy=(x_axis[1],delay_drop[1]), xytext=(8,10),arrowprops=dict(
        arrowstyle = '->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[5], delay_return[5]), xytext=(3, 30), arrowprops=dict(
        arrowstyle='->'
    ))

    come_p = 0.04
    for i in range(0, x_axis.shape[0]):
        delay_drop[i] = drop_model.get_avg_delay(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
        delay_return[i] = return_model.get_avg_delay(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    # axe_1.plot(x_axis, delay_drop, linewidth=0.5, color="black", marker="*",
    #            label=r"$\lambda_1=%s$ Model Drop" % serve_p)
    # axe_1.plot(x_axis, delay_return, linewidth=0.5, color="gray", marker="*",
    #            label=r"$\lambda_1=%s$ Model Return" % serve_p)
    axe_1.plot(x_axis, delay_drop, linewidth=0.5, color="black", marker="*",
               label=r"$\lambda_1=%s$" % come_p)
    axe_1.plot(x_axis, delay_return, linewidth=0.5, color="gray", marker="*",)

    axe_1.annotate("Model Drop", xy=(x_axis[3], delay_drop[3]), xytext=(8, 10), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[4], delay_return[4]), xytext=(3, 30), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.set_xlabel("cache k")
    axe_1.set_xlim(left=1)
    axe_1.set_xlim(right=10)
    axe_1.set_xticks(range(1, x_axis.shape[0] + 1, 1))
    axe_1.set_ylabel(r"avg_delay $\zeta$")
    axe_1.legend()
    # axe_1.set_title(r'$\zeta$ - k')
    fig.show(warn=False)
    fig.savefig(fname="./result/delay_compare.svg", format="svg")


def compare_blocking_ratio():
    # #####################compare block_ratio######################
    x_axis = np.arange(1, 11, 1)
    block_drop_1 = np.arange(1, 11, 1, dtype="float")
    block_return_1 = np.arange(1, 11, 1, dtype="float")
    block_drop_2 = np.arange(1, 11, 1, dtype="float")
    block_return_2 = np.arange(1, 11, 1, dtype="float")

    for i in range(0, x_axis.shape[0]):
        block_drop_1[i] = drop_model.get_block_ratio(0.16, come_s, serve_p, serve_s, x_axis[i], nc)
        block_return_1[i] = return_model.get_block_ratio(0.16, come_s, serve_p, serve_s, x_axis[i], nc)
        block_drop_2[i] = drop_model.get_block_ratio(0.04, come_s, serve_p, serve_s, x_axis[i], nc)
        block_return_2[i] = return_model.get_block_ratio(0.04, come_s, serve_p, serve_s, x_axis[i], nc)

    fig = plt.figure(figsize=(7.00, 5.25), clear=True)
    axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axe_1.tick_params(top='on', right='on', direction='in')
    # axe_1.plot(x_axis, block_drop_1, label=r"$\lambda_1=0.16$ Model Drop", marker=".", color="black",linewidth=0.5)
    # axe_1.plot(x_axis, block_return_1, label=r"$\lambda_1=0.16$ Model Return", marker=".", color="gray",linewidth=0.5)
    # axe_1.plot(x_axis, block_drop_2, label=r"$\lambda_1=0.04$ Model Drop", marker="*", color="black",linewidth=0.5)
    # axe_1.plot(x_axis, block_return_2, label=r"$\lambda_1=0.04$ Model Return", marker="*", color="gray",linewidth=0.5)
    axe_1.plot(x_axis, block_drop_1, label=r"$\lambda_1=0.16$", marker=".", color="black",linewidth=0.5)
    axe_1.plot(x_axis, block_return_1, marker=".", color="gray",linewidth=0.5)
    axe_1.plot(x_axis, block_drop_2, label=r"$\lambda_1=0.04$", marker="*", color="black",linewidth=0.5)
    axe_1.plot(x_axis, block_return_2, marker="*", color="gray",linewidth=0.5)
    axe_1.annotate("Model Drop", xy=(x_axis[1],block_drop_1[1]),xytext=(6,0.14), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Drop", xy=(x_axis[2], block_drop_2[2]), xytext=(6, 0.14), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[5], block_return_1[5]), xytext=(8, 0.12), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[6], block_return_2[6]), xytext=(8, 0.12), arrowprops=dict(
        arrowstyle='->'
    ))

    axe_1.set_xlabel("cache k")
    axe_1.set_xlim(left=1)
    axe_1.set_xlim(right=10)
    axe_1.set_xticks(range(1, x_axis.shape[0] + 1, 1))
    # axe_1.set_ylabel("block_ratio")
    axe_1.set_ylabel(r'block_ratio $\beta$')
    axe_1.legend()
    # axe_1.set_title(r'$\beta$ - k')
    fig.show(warn=False)
    fig.savefig(fname="./result/block_ratio_compare.svg", format="svg")


def compare_loss_ratio():
    # #####################compare interrupt_ratio######################
    x_axis = np.arange(1, 11, 1)
    interrupt_drop = np.arange(1, 11, 1, dtype="float")
    interrupt_return = np.arange(1, 11, 1, dtype="float")
    fig = plt.figure(figsize=(7.00, 5.25), clear=True)
    axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axe_1.tick_params(top='on', right='on', direction='in')
    come_p = 0.16
    for i in range(0, x_axis.shape[0]):
        interrupt_drop[i] = drop_model.get_interrupt_ratio(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
        interrupt_return[i] = return_model.get_interrupt_ratio(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    # axe_1.plot(x_axis, interrupt_drop, label=r"$\lambda_1=%s$ Model Drop" % come_p, marker=".",color="black", linewidth=0.5)
    # axe_1.plot(x_axis, interrupt_return,label=r"$\lambda_1=%s$ Model Return" % come_p, marker=".",color="gray", linewidth=0.5)
    axe_1.plot(x_axis, interrupt_drop, label=r"$\lambda_1=%s$" % come_p, marker=".", color="black",
               linewidth=0.5)
    axe_1.plot(x_axis, interrupt_return, marker=".", color="gray",
               linewidth=0.5)
    axe_1.annotate("Model Drop", xy=(x_axis[1],interrupt_drop[1]),xytext=(3,0.015),arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[5],interrupt_return[5]),xytext=(8,0.015),arrowprops=dict(
        arrowstyle='->'
    ))
    come_p = 0.04
    for i in range(0, x_axis.shape[0]):
        interrupt_drop[i] = drop_model.get_interrupt_ratio(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
        interrupt_return[i] = return_model.get_interrupt_ratio(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    # axe_1.plot(x_axis, interrupt_drop, label=r"$\lambda_1=%s$ Model Drop" % come_p, marker="*",color="black", linewidth=0.5)
    # axe_1.plot(x_axis, interrupt_return, label=r"$\lambda_1=%s$ Model Return" % come_p,marker="*",color="gray", linewidth=0.5)
    axe_1.plot(x_axis, interrupt_drop, label=r"$\lambda_1=%s$" % come_p, marker="*", color="black",
               linewidth=0.5)
    axe_1.plot(x_axis, interrupt_return, marker="*", color="gray",
               linewidth=0.5)
    axe_1.annotate("Model Drop", xy=(x_axis[1], interrupt_drop[1]), xytext=(3, 0.015), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[5], interrupt_return[5]), xytext=(8, 0.015), arrowprops=dict(
        arrowstyle='->'
    ))



    axe_1.set_xlabel("cache k")
    axe_1.set_xlim(left=1)
    axe_1.set_xlim(right=10)
    axe_1.set_xticks(range(1, x_axis.shape[0] + 1, 1))
    axe_1.set_ylabel(r'interrupt_ratio $\gamma$')
    # axe_1.grid()
    axe_1.legend()
    # axe_1.set_title(r'$\gamma$ - k')
    fig.show(warn=False)
    fig.savefig(fname="./result/interrupt_ratio_compare.svg", format="svg")


def compare_throughput():
    # #####################compare io######################
    x_axis = np.arange(1, 11, 1)
    io_drop = np.arange(1, 11, 1, dtype="float")
    io_return = np.arange(1, 11, 1, dtype="float")

    fig = plt.figure(figsize=(7.00, 5.25), clear=True)
    axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axe_1.tick_params(top='on', right='on', direction='in')
    come_p = 0.16
    for i in range(0, x_axis.shape[0]):
        io_drop[i] = drop_model.get_throughput_capcity(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
        io_return[i] = return_model.get_throughput_capcity(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    # axe_1.plot(x_axis, io_drop, marker=".", label=r"$\lambda_1=%s$ Model Drop" % come_p, color="black", linewidth=0.5)
    # axe_1.plot(x_axis, io_return, marker=".",label=r"$\lambda_1=%s$ Model Return" % come_p, color="gray", linewidth=0.5)
    axe_1.plot(x_axis, io_drop, marker=".", label=r"$\lambda_1=%s$" % come_p, color="black", linewidth=0.5)
    axe_1.plot(x_axis, io_return, marker=".", color="gray", linewidth=0.5)
    axe_1.annotate("Model Drop", xy=(x_axis[1],io_drop[1]),xytext=(4,0.20),arrowprops=dict(
        arrowstyle = '->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[5], io_return[5]), xytext=(7, 0.20), arrowprops=dict(
        arrowstyle='->'
    ))
    come_p = 0.04
    for i in range(0, x_axis.shape[0]):
        io_drop[i] = drop_model.get_throughput_capcity(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
        io_return[i] = return_model.get_throughput_capcity(come_p, come_s, serve_p, serve_s, x_axis[i], nc)
    # axe_1.plot(x_axis, io_drop, marker="*", label=r"$\lambda_1=%s$ Model Drop" %come_p, color="black", linewidth=0.5)
    # axe_1.plot(x_axis, io_return, marker="*",label=r"$\lambda_1=%s$ Model Return" %come_p, color="gray", linewidth=0.5)
    axe_1.plot(x_axis, io_drop, marker="*", label=r"$\lambda_1=%s$" % come_p, color="black", linewidth=0.5)
    axe_1.plot(x_axis, io_return, marker="*", color="gray", linewidth=0.5)
    axe_1.annotate("Model Drop", xy=(x_axis[1], io_drop[1]), xytext=(4, 0.20), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.annotate("Model Return", xy=(x_axis[5], io_return[5]), xytext=(7, 0.20), arrowprops=dict(
        arrowstyle='->'
    ))
    axe_1.set_xlabel("cache k")
    axe_1.set_xlim(left=1)
    axe_1.set_xlim(right=10)
    axe_1.set_xticks(range(1, x_axis.shape[0] + 1, 1))
    axe_1.set_ylabel(r"throughput_capcity $\theta$")
    axe_1.legend()
    # axe_1.set_title(r'$\theta$ - k')
    fig.show(warn=False)
    fig.savefig(fname="./result/io_compare.svg", format="svg")


def compare_status():
    # ######## count possibility by status #########
    k = 5
    staitc_matrix = drop_model.get_static_matrix(come_p, come_s, serve_p, serve_s, k, nc)
    staitc_matrix_2 = return_model.get_static_matrix(come_p, come_s, serve_p, serve_s, k, nc)
    fig = plt.figure(figsize=(7.00, 5.25), clear=True)
    axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    x_axis = range(0, 2 * k + 3)
    y_axis = range(0, 2 * k + 3)
    y_axis_2 = range(0, 2 * k + 3)
    for i in range(len(x_axis)):
        y_axis[i] = staitc_matrix[0, i]
        y_axis_2[i] = staitc_matrix_2[0, i]
    axe_1.bar(x_axis, y_axis, label="drop_model")
    axe_1.bar(x_axis, y_axis_2, label="return_model")
    axe_1.legend()
    axe_1.set_title("distribution")
    axe_1.set_xlabel("status")
    axe_1.set_ylabel("possibility")
    fig.show()
    fig.savefig("./result/distribution.svg", format="svg")


def plot_serve_s_m():
    # #########change of serve_s_m
    fig = plt.figure(figsize=(7.00, 5.25), clear=True)
    axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    k = 10
    nc = 5
    x = range(1, k + 1, 1)
    y = range(1, k + 1, 1)
    y_1 = range(1, k + 1, 1)
    y_2 = range(1, k + 1, 1)
    for m in range(0, len(x), 1):
        serve_s_m = math.ceil(m * 5 / (k + 1.0)) * serve_s
        serve_s_m_1 = math.ceil(m * 10 / (k + 1.0)) * serve_s
        serve_s_m_2 = math.ceil(m * 15 / (k + 1.0)) * serve_s
        y[m] = serve_s_m
        y_1[m] = serve_s_m_1
        y_2[m] = serve_s_m_2
    axe_1.plot(x, y, label="Nc = 5")
    axe_1.plot(x, y_1, label="Nc = 10")
    axe_1.plot(x, y_2, label="Nc = 15")
    axe_1.set_xlim(left=1)
    axe_1.set_xlim(right=10)
    axe_1.set_xlabel("m")
    axe_1.set_xticks(range(1, len(x) + 1, 1))
    axe_1.set_ylabel(r'$\vartheta_m$')
    axe_1.set_title(r'$\vartheta_m$ - m')
    axe_1.legend()
    fig.show()
    fig.savefig("./result/serve_s_m.svg", format="svg")


compare_delay()
compare_blocking_ratio()
compare_throughput()
compare_loss_ratio()
