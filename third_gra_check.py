# encoding:utf-8
import numpy as np
from testCases import *
from gc_utils import *

# def forward_linear(x, para):
#     A = para * x
#     return A
# def backword_linear(x, para):
#     return x
#
# # the para in here is a scalar, but in the NN para is a dict, so we need to change to vectors
# def gradient_check(x, theat, epsilon = 1e-7):
#     theatplus = theat + epsilon
#     theatminus = theat - epsilon
#
#     # compute approx
#     f1 = forward_linear(x, theatplus)   # 返回的是一个向量
#     f2 = forward_linear(x, theatminus)
#     gradapprox = (f1-f2) / (2*epsilon)
#
#     grad = backword_linear(x, theat)
#
#     numerator = np.linalg.norm(grad - gradapprox)
#     denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
#     difference = numerator / denominator
#     if difference < 1e-7:
#         print("Correct.")
#     else:
#         print("Error,")
#     return difference
#
#
# # def initialize_parameters(layer_dims):
# #
# #     np.random.seed(3)
# #     parameters = {}
# #     L = len(layer_dims)  # number of layers in the network
# #
# #     for l in range(1, L):
# #         parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
# #         parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
# #
# #         assert (parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l - 1])
# #         assert (parameters['W' + str(l)].shape == layer_dims[l], 1)
# #
# #     return parameters
#
def forward_propagation_three(X, Y, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    m = X.shape[1]
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)
    return cost, cache
#
#
def backward_propagation_three(X, Y, cache):

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
#
# def gradient_check_three(grads, para, x, y, epsilon = 1e-7):
#     # para in here should be vectors
#     para, _= dictionary_to_vector(parameters=para)
#     # grads is the output of the backward_propagation_three, is a dict
#     grad = gradients_to_vector(grads)
#     theatplus = para + epsilon
#     theatminus = para - epsilon
#
#     # compute approx
#     f1, _ = forward_propagation_three(x, y, vector_to_dictionary(theatplus))
#     f2, _ = forward_propagation_three(x, y, vector_to_dictionary(theatminus))
#     gradapprox = (f1-f2) / (2*epsilon)
#
#
#     numerator = np.linalg.norm(grad - gradapprox)
#     denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
#     difference = numerator / denominator
#     if difference < 1e-7:
#         print("Correct.")
#     else:
#         print("Error,", difference)
#     return difference
#
def gradient_check_three(grads, para, x, y, epsilon = 1e-7):
    # para in here should be vectors
    para, _= dictionary_to_vector(parameters=para)
    # grads is the output of the backward_propagation_three, is a dict
    grad = gradients_to_vector(grads)
    num_para = para.shape[0]
    J_plus = np.zeros((num_para, 1))
    J_mins = np.zeros((num_para, 1))
    gradapprox = np.zeros((num_para, 1))
    # 计算大概值的时候，每次只改变一个参数，w11其他参数不变，去计算J_plus[i]和J_mins[i]
    for i in range(num_para):
        theatplus = np.copy(para)
        theatplus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_three(x, y, vector_to_dictionary(theatplus))

        # compute J_minus
        theatminus = np.copy(para)
        theatminus[i][0] -= epsilon
        J_mins[i], _ = forward_propagation_three(x, y, vector_to_dictionary(theatminus))
        gradapprox[i] = (J_plus[i]-J_mins[i]) / (2*epsilon)

    # 将计算的大概值和cost对所有参数的梯度下降进行比较
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = numerator / denominator
    if difference < 2e-7:
        print("Correct.", difference)
    else:
        print("Error,", difference)
    return difference
#
def main():
    # test one: linear model y = 4x to check gd
    # x, para = 2, 4
    # # A = forward_linear(x, para)
    # # delt = backword_linear(x, para)
    # diff = gradient_check(x, para)

    # the para in here is a scalar, but in the NN para is a dict, so we need to

    # print(diff)

    # test two: NN of three layers
    X, Y, para = gradient_check_n_test_case()
    cost, cache = forward_propagation_three(X, Y, para)
    grad = backward_propagation_three(X, Y, cache)
    diff = gradient_check_three(grad, para, X, Y)

    pass

if __name__ == '__main__':
    main()
# GRADED FUNCTION: gradient_check_n
# def gradient_check_three(parameters, gradients, X, Y, epsilon=1e-7):
#     """
#     Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
#
#     Arguments:
#     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
#     grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
#     x -- input datapoint, of shape (input size, 1)
#     y -- true "label"
#     epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
#
#     Returns:
#     difference -- difference (2) between the approximated gradient and the backward propagation gradient
#     """
#
#     # Set-up variables
#     parameters_values, _ = dictionary_to_vector(parameters)
#     grad = gradients_to_vector(gradients)
#     num_parameters = parameters_values.shape[0]
#     J_plus = np.zeros((num_parameters, 1))
#     J_minus = np.zeros((num_parameters, 1))
#     gradapprox = np.zeros((num_parameters, 1))
#     # Compute gradapprox
#     for i in range(num_parameters):
#         # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
#         # "_" is used because the function you have to outputs two parameters but we only care about the first one
#         ### START CODE HERE ### (approx. 3 lines)
#         thetaplus = np.copy(parameters_values)  # Step 1
#         thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
#         J_plus[i], _ = forward_propagation_three(X, Y, vector_to_dictionary(thetaplus))  # Step 3
#         ### END CODE HERE ###
#
#         # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
#         ### START CODE HERE ### (approx. 3 lines)
#         thetaminus = np.copy(parameters_values)  # Step 1
#         thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
#         J_minus[i], _ = forward_propagation_three(X, Y, vector_to_dictionary(thetaminus))  # Step 3
#         ### END CODE HERE ###
#
#         # Compute gradapprox[i]
#         ### START CODE HERE ### (approx. 1 line)
#         gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
#         ### END CODE HERE ###
#
#     # Compare gradapprox to backward propagation gradients by computing difference.
#     ### START CODE HERE ### (approx. 1 line)
#     numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
#     denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
#     difference = numerator / denominator  # Step 3'
#     ### END CODE HERE ###
#     if difference > 1e-7:
#         print(
#             "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
#     else:
#         print(
#             "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
#
#     return difference
#

# X, Y, para = gradient_check_n_test_case()
# cost, cache = forward_propagation_three(X, Y, para)
# grad = backward_propagation_three(X, Y, cache)
# diff = gradient_check_three(para, grad, X, Y)