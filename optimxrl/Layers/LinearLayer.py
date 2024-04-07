import numpy as np  # import numpy library
from ..utils import (
    initialize_parameters,
    safe_div,
)  # import function to initialize weights and biases


class LinearLayer:
    """
    This Class implements all functions to be executed by a linear layer
    in a computational graph

    Args:
        input_shape: input shape of Data/Activations
        n_out: number of neurons in layer
        ini_type: initialization type for weight parameters, default is "plain"
                  Opitons are: plain, xavier and he

    Methods:
        forward(A_prev)
        backward(upstream_grad)
        update_params(learning_rate)

    """

    def __init__(self, n_in, n_out, ini_type="plain", opt="adam"):
        """
        The constructor of the LinearLayer takes the following parameters

        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
        """
        self.layer_type = "Linear"
        self.n_in = n_in
        self.n_out = n_out
        self.m = 1  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        self.params = initialize_parameters(
            n_in, n_out, ini_type
        )  # initialize weights and bias
        self.Z = np.zeros(
            (self.params["W"].shape[0], 1)
        )  # create space for resultant Z output
        self.eps = 1e-10
        # momentum
        # self.m1 = np.random.uniform(0.1, 1, self.w.shape)
        # self.m2 = np.random.uniform(0.1, 1, self.w.shape)
        self.b1 = 0.9  # Adam, if b1 = 0. -> Adam = RMSprop
        self.b2 = 0.99
        self.opt = opt
        self.eta = 3e-4  # 1e-3

    def forward(self, A_prev, W, b, predict=True):
        """
        This function performs the forwards propagation using activations from previous layer

        Args:
            A_prev:  Activations/Input Data coming into the layer from previous layer
        """

        # self.A_prev = A_prev  # store the Activations/Training Data coming in
        Z = np.dot(W, A_prev) + b  # compute the linear function
        if predict:
            return Z
        else:
            return A_prev, Z

    def backward(self, upstream_grad, A_prev, W):
        """
        This function performs the back propagation using upstream gradients

        Args:
            upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """

        # derivative of Cost w.r.t W
        dW = np.dot(upstream_grad, A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        dA_prev = np.dot(W.T, upstream_grad)
        return dA_prev, dW, db

    def update_params(
        self, dW, db, W, b, m1=None, m2=None, model_updated_cnt=None, lr=0.1
    ):
        """
        This function performs the gradient descent update

        Args:
            learning_rate: learning rate hyper-param for gradient descent, default 0.1
            adam's lr: self.eta
        """
        if self.opt in ["adam", "rmsprop"]:  # RMSprop
            # print(m1["W"].shape, dW.shape)
            m1["W"] = self.b1 * m1["W"] + (1 - self.b1) * dW
            m1["b"] = self.b1 * m1["b"] + (1 - self.b1) * db
            m2["W"] = self.b2 * m2["W"] + (1 - self.b2) * dW**2
            m2["b"] = self.b2 * m2["b"] + (1 - self.b2) * db**2
            if self.opt == "rmsprop":
                W += self.eta * dW / (np.sqrt(m2["W"]) + self.eps)
                b += self.eta * db / (np.sqrt(m2["b"]) + self.eps)
            elif self.opt == "adam":
                m_b_hat = m1["b"] / (
                    1.0 - self.b1 ** (model_updated_cnt + 1)
                )  # bias correction
                v_b_hat = m2["b"] / (
                    1.0 - self.b2 ** (model_updated_cnt + 1)
                )  # bias correction
                m_w_hat = m1["W"] / (
                    1.0 - self.b1 ** (model_updated_cnt + 1)
                )  # bias correction
                v_w_hat = m2["W"] / (
                    1.0 - self.b2 ** (model_updated_cnt + 1)
                )  # bias correction
                # print(v_b_hat,m2["b"],"v_b_hat")
                W -= self.eta * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
                b -= self.eta * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
                # W += self.eta * m1["W"] / (np.sqrt(m2["W"]) + self.eps)
                # W += self.eta * m1["W"] / (np.sqrt(m2["W"]) + self.eps)
                # print(b,m1["b"] ,m2["b"],"llll")
                # b += self.eta * m1["b"] / (np.sqrt(m2["b"]) + self.eps)
            else:
                raise ValueError(f"{self.opt} is not supported.")
        else:  # normal
            W = W - lr * dW  # update weights
            b = b - lr * db  # update bias(es)
        if self.opt in ["adam", "rmsprop"]:
            return W, b, m1, m2
        else:
            return W, b
