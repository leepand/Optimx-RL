import numpy as np

###################### regularization ####################################################
def L2_norm(lam, a):
    """
    2-Norm regularizer
    """
    return lam * a


############################ MLP (Multi-layer perceptron neural networks) ###########################################
### network layer
class MLPLayer:
    """
    actiavtion layer for model building:
        layer(input_dimension,number_of_nodes)

    parameters:
        f   : activation function
        w   : weights

        reg : regularizer function
        lam : regularizer lambda
        eta : learning rate

        opt : optimizer ('Adam','RMSprop','normal')
        eps : "don't devide by zero!!"
        b1  : momentumparameter for 'Adam' optimizer
        b2  : momentumparameter for 'RMSprop' and 'Adam' optimizer
        m1  : momentum for 'Adam' optimizer
        m2  : momentum for 'RMSprop' and 'Adam' optimizer

        count: number of updates

    """

    def __init__(self, in_dim, nodes=32, no_bias=False, activation_fun=None):

        # activation and weights
        self.no_bias = no_bias
        self.nodes = nodes
        self.in_dim = in_dim
        self.f = activation_fun  # f_tanh
        # by default Xavier init
        # np.random.randn(nodes, in_dim) / np.sqrt(in_dim)#
        # np.random.randn(nodes, in_dim+1) / np.sqrt(in_dim+1)#
        if self.no_bias:
            self.w = np.random.uniform(-0.1, 0.1, (nodes, in_dim))
        else:
            self.w = np.random.uniform(-0.1, 0.1, (nodes, in_dim + 1))
        ### Xavier init:
        # w = np.random.randn(neurons, input_dimension) / np.sqrt(input_dimension)

        # momentum
        self.m1 = np.random.uniform(0.1, 1, self.w.shape)
        self.m2 = np.random.uniform(0.1, 1, self.w.shape)
        self.b1 = 0.9  # Adam, if b1 = 0. -> Adam = RMSprop
        self.b2 = 0.99
        self.opt = "Adam"
        self.eps = 1e-10

        # regularizer
        self.reg = L2_norm
        self.lam = 0.0  # 1e-8

        # learning
        self.count = 0
        self.eta = 1e-4

    def forward(self, input_, w=None):
        """
        forward pass (computes activation)
            return: activation(input * weights[+ bias])
        """
        ##### IF no_bias != True  :
        if w is not None:
            self.w = w

        if self.no_bias:
            x1 = input_
        else:
            x1 = np.vstack((input_.T, np.ones(input_.shape[0]))).T
        # print('fw x1: ',self.x1.shape)
        # self.h1 = np.dot(self.x1, self.w.T).T
        h1 = np.dot(x1, self.w.T).T
        # print('fw h1: ',self.h1.shape)
        # self.s = self.f(self.h1)
        s = self.f(h1)
        # print('fw s: ',self.s.shape)
        return s, h1, x1

    def backward(self, L_error, x1, h1, w=None):
        """
        backward pass (computes gradient)
            return: layer delta
        """
        # print('L_error :  ',L_error.shape)
        if w is not None:
            self.w = w
        L_grad = L_error * self.f(h1, True).T
        # print('L_grad :  ',self.L_grad.shape)
        delta_W = -1.0 / (x1).shape[0] * np.dot(L_grad.T, x1) - self.reg(
            self.lam, self.w
        )
        if self.no_bias:
            return np.dot(self.w.T, L_grad.T).T, delta_W
        else:
            return np.dot(self.w.T[1:], L_grad.T).T, delta_W

    def update(self, m1=None, m2=None, w=None, delta_W=None):
        """
        update step (updates weights & momentum)
        """
        if m1 is not None:
            self.m1 = m1
            self.m2 = m2
            self.w = w
            self.delta_W = delta_W

        self.m1 = self.b1 * self.m1 + (1 - self.b1) * self.delta_W
        self.m2 = self.b2 * self.m2 + (1 - self.b2) * self.delta_W**2
        if self.opt == "RMSprop":
            self.w += self.eta * self.delta_W / (np.sqrt(self.m2) + self.eps)
        if self.opt == "Adam":
            self.w += self.eta * self.m1 / (np.sqrt(self.m2) + self.eps)
        if self.opt == "normal":
            self.w += self.eta * self.delta_W
        self.count += 1
        m1 = self.m1
        m2 = self.m2
        w = self.w
        return w, m1, m2

    def reset(self):
        """
        weights & momentum reset
        """
        self.w = np.random.uniform(-0.7, 0.7, (self.nodes, self.in_dim + 1))
        self.m1 = np.random.uniform(0.0, 1, self.w.shape)
        self.m2 = np.random.uniform(0.0, 1, self.w.shape)
