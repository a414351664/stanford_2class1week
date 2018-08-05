# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import *
'''
随机初始化×10    0.83,　0.86
随机初始化       0.99,　0.96
he初始化        0.99   0.96
'''
# Three kinds of ways to initialize the weights and bias
def initial_zeros(layers_dims):
    np.random.seed(3)
    para = {}
    for i in range(1, len(layers_dims)):
        para["W" + str(i)] = np.zeros((layers_dims[i], layers_dims[i-1]))
        para["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert (para["W" + str(i)].shape == (layers_dims[i], layers_dims[i-1]))
        assert (para["b" + str(i)].shape == (layers_dims[i], 1))
    return para
def initial_random(layers_dims):
    np.random.seed(3)
    para = {}
    for i in range(1, len(layers_dims)):
        para["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])
        para["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert (para["W" + str(i)].shape == (layers_dims[i], layers_dims[i-1]))
        assert (para["b" + str(i)].shape == (layers_dims[i], 1))
    return para
def initial_he(layers_dims):
    np.random.seed(3)
    para = {}
    for i in range(1, len(layers_dims)):
        para["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2.0/layers_dims[i - 1])
        para["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert (para["W" + str(i)].shape == (layers_dims[i], layers_dims[i-1]))
        assert (para["b" + str(i)].shape == (layers_dims[i], 1))
    return para

def three_layer(x, y, learning_rate = 0.01, iter_num = 15000, initialization = ""):
    # define the parameter, cost
    para = {}
    costs = []
    # 1.initialize
    layers_dims = (x.shape[0], 10, 5, 1)
    if initialization == "zeros":
        para = initial_zeros(layers_dims)
    if initialization == "random":
        para = initial_random(layers_dims)
    if initialization == "he":
        para = initial_he(layers_dims)

    for i in range(iter_num):
        # 2.compute forward
        A3, cache = forward_propagation(x, para)

        # 3.compute cost
        cost = compute_loss(A3, y)

        # 4.backword
        grads = backward_propagation(x, y, cache)

        # 5.updata
        para = update_parameters(para, grads, learning_rate)
        if i % 100 == 0:
            print("%i times,", i,  "cost is %f", cost)
            costs.append(cost)
    return para, costs

def main():
    plt.rcParams["figure.figsize"] = (7.0, 4.0)
    plt.rcParams["image.interpolation"] = 'nearest'
    plt.rcParams["image.cmap"] = "gray"
    # (nx, m)   (1, m)
    train_X, train_Y, test_X, test_Y = load_dataset()
    # plt.show() # 显示两个大圆圈
    # 使用不同的初始化构建一个三层的神经网络

    initialization = "random"

    para, costs = three_layer(train_X, train_Y, initialization=initialization)

    plt.subplot(211)
    # 获得当前子图
    axes_1 = plt.gca()
    axes_1.plot(np.squeeze(costs))
    axes_1.set_xlabel("times per tens")
    axes_1.set_ylabel("The loss")
    axes_1.set_title("initialization is " + str(initialization))
    pre_train = predict(train_X, train_Y, para)
    pre_test = predict(test_X, test_Y, para)
    print('The accuracy of train is', pre_train, '\nThe test is', pre_test)

    plt.subplot(212)    # 其实也可以直接axes = plt.subplot(212)
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    axes.set_xlabel("x1_feature")
    axes.set_xlabel("x2_feature")
    axes.set_title("decisoin_boundray")
    # axes_1.set_xlabel("times") # 对子图进行操作
    plot_decision_boundary(lambda x:predict_dec(para, x.T), test_X, test_Y)
    plt.show()
    pass


if __name__ == '__main__':
    main()