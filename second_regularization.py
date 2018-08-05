# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from reg_utils import  *
from testCases import *
'''
无lambd 和 dropout learning_rate = 0.03, iter = 15000
train Accuracy: 0.933649289099526
test Accuracy: 0.945

无lambd 和 dropout learning_rate = 0.3, iter = 15000
train Accuracy: 0.94
test Accuracy: 0.915
'''

def three_layer(x, y, learning_rate = 0.3, iter_num = 30000, lambd = 0.0, keep_prob = 1.0):
    # define the parameter, cost
    para = {}
    costs = []
    # 1.initialize
    layers_dims = (x.shape[0], 20, 3, 1)
    para = initialize_parameters(layers_dims)

    for i in range(iter_num):
        # 2.compute forward
        # first, u should judge if keep_pro == 1
        if keep_prob == 1.0:
            A3, cache = forward_propagation(x, para)
        elif keep_prob < 1.0:
            A3, cache = forward_propagation_with_dropout(x, para, keep_prob)
            # print("forward_propagation_with_dropout")
        # X_a, paraaa = forward_propagation_with_dropout_test_case()
        # A33, CACHE = forward_propagation_with_dropout(X_a, paraaa, keep_prob)
        # print(A33)
        # 3.compute cost
        # first, u should judge if lambd == 0
        if lambd == 0:
            cost = compute_cost(A3, y)
        elif lambd > 0.0:
            cost = compute_cost_with_regularization(A3, y, para, lambd)
            # print("compute_cost_with_regularization.")

        # 4.backword
        if keep_prob == 1.0 and lambd == 0.0:
            grads = backward_propagation(x, y, cache)
        elif keep_prob < 1.0 and lambd == 0.0:
            grads = backward_propagation_with_dropout(x, y, cache, keep_prob)
            # print("backward_propagation_with_dropout")
        else:
            grads = backward_propagation_with_regularization(x, y, cache, para, lambd)
            # print("backward_propagation_with_regularization")

        # 5.updata
        para = update_parameters(para, grads, learning_rate)
        if i % 1000 == 0:
            print("%i times,", i,  "cost is %f", cost)
            costs.append(cost)
    return para, costs

def main():
    plt.rcParams["figure.figsize"] = (7.0, 4.0)
    plt.rcParams["image.interpolation"] = 'nearest'
    plt.rcParams["image.cmap"] = "gray"
    # (nx, m)2,211   (1, m)
    # test has 200 examples
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    # plt.show()

    # lambd = 0.0
    # keep_prob = 1.0
    lambd = 0.0
    keep_prob = 0.86
    para, costs = three_layer(train_X, train_Y, lambd=lambd, keep_prob=keep_prob)
    #
    plt.subplot(211)
    # 获得当前子图
    axes_1 = plt.gca()
    axes_1.plot(np.squeeze(costs))
    axes_1.set_xlabel("times per tens")
    axes_1.set_ylabel("The loss")
    axes_1.set_title("keep_prob is "+str(keep_prob)+"lambd is " + str(lambd))
    pre_train = predict(train_X, train_Y, para)
    pre_test = predict(test_X, test_Y, para)
    print('The accuracy of train is', pre_train, '\nThe test is', pre_test)
    #
    plt.subplot(212)    # 其实也可以直接axes = plt.subplot(212)
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.4])
    axes.set_ylim([-0.75, 0.65])
    axes.set_xlabel("x1_feature")
    axes.set_xlabel("x2_feature")
    axes.set_title("decisoin_boundray")
    # axes_1.set_xlabel("times") # 对子图进行操作
    plot_decision_boundary(lambda x:predict_dec(para, x.T), train_X, train_Y)
    plt.show()
    pass


if __name__ == '__main__':
    main()