import numpy as np
from ..ActivationFunctions.Tanh import Tanh2

###################### loss ####################################################
def ce(y, yt, dev=False):  ############ not robust !!
    """
    cross entropy
    argmax stuff maybe otherwise near zero logs are silly
    """
    if dev == True:
        return yt - y
    loss = [(yt[i]).dot(np.log(y[i])) for i in range(y.shape[0])]
    loss = np.array(loss) * -1
    return np.sum(loss) / (loss.shape[0] * 1.0)


def bce(ya, yta, dev=False):  ############ not robust !!
    """
    binary cross entropy
    """
    if dev == True:
        return (yta - ya) / ((1 - yta) * yta)
    return -(
        np.sum(ya * np.log(yta) + (1.0 - yta) * np.log(1.0 - yta))
        / (yta.shape[0] * 2.0)
    )


def qef(ya, yta, dev=False):
    """
    quadratic error function ||prediction-target||²
    """
    if dev == True:
        return yta - ya
    return np.sum((yta - ya) ** 2) / (yta.shape[0] * 2.0)


def phl(y, yt, dev=False, delta=1.0):
    """
    subquadratic error function (pseudo huber loss)
    """
    a = yt - y
    if dev == True:
        return a / (np.sqrt(a**2 / delta**2 + 1))
    return np.sum(
        (delta**2) * (np.sqrt(1 + (a / delta) ** 2) - 1) / (yt.shape[0] * 2.0)
    )


###################### regularization ####################################################
def L2_norm(lam, a):
    """
    2-Norm regularizer
    """
    return lam * a


def L1_norm(lam, a):
    """
    1-Norm regularizer
    """
    return lam * np.sign(a)


###################### activation  ####################################################
def f_elu(a, dev=False):
    """
    exponential linear unit
        ~softplus [0,a]
    """
    if dev:
        return np.where(a >= 0.0, f_elu(a) + a, 1)
    return np.where(a >= 0.0, a * (np.exp(a) - 1), a)


def f_softmax(a, dev=False):
    """
    softmax transfer function
        sigmoidal [0,1]
    """

    if dev == True:
        return f_softmax(a) * (1 - f_softmax(a))
    return np.exp(a) / np.sum(np.exp(a))


def f_lgtr(a, dev=False):
    """
    (robust) logistic transfer function
        sigmoidal [0,1]
    """
    if dev == True:
        return (1 - np.tanh(a / 2.0) ** 2) / 2.0
    return (np.tanh(a / 2.0) + 1) / 2.0


def f_stoch(a, dev=False):
    """
    stochastic transfer function
        activates if activated input > ~Uniform
        binary [0,1]
    """
    if dev == True:
        return np.zeros(a.shape)
    x = f_lgtr(a, dev=False)
    rand = np.random.random(x.shape)
    return np.where(rand < x, 1, 0)


def f_tanh(a, dev=False):
    """
    hyperbolic tangent transfer function
        sigmoidal [-1,1]
    """
    if dev == True:
        return 1 - np.tanh(a) ** 2
    return np.tanh(a)


def f_atan(a, dev=False):
    """
    arcus tangent transfer function
        sigmoidal [-pi/2, pi/2]
    """
    if dev == True:
        return 1 / (a**2 + 1)
    return np.arctan(a)


def f_sp(a, dev=False):
    """
    softplus transfer function
        [0,a]

        ### kinda clip it...to make more robust
    """
    if dev == True:
        return np.exp(a) / (np.exp(a) + 1.0)
    return np.log(np.exp(a) + 1.0)


def f_relu(a, dev=False):
    """
    rectified linear transfer function
        [0,a]
    """
    if dev == True:
        return np.maximum(0, np.sign(a))
    return np.maximum(0.0, a)


def f_leaky(a, dev=False, leak=0.01):
    """
    leaky rectified linear transfer function
         [-leaky*a,a]
    """
    if dev == True:
        signs = np.sign(a)

        return np.where(signs > 0.0, signs, 0.01 * signs)
    return np.where(a > 0.0, a, leak * a)


def f_bi(a, dev=False):
    """
    bent identity transfer function
    """
    if dev == True:
        return a / (2.0 * np.sqrt(a**2 + 1)) + 1
    return (np.sqrt(a**2 + 1) - 1) / 2.0 + a


def f_iden(a, dev=False):
    """
    identity transfer function
    """
    if dev == True:
        return np.ones(a.shape)
    return a


def f_bin(a, dev=False):
    """
    binary step transfer function
    """
    if dev == True:
        return np.zeros(a.shape)
    return np.sign(f_relu(a))


############################ MLP (Multi-layer perceptron neural networks) ###########################################
### network layer
class layer:
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

    def __init__(
        self, in_dim, nodes=32, no_bias=False, opt="Adam", activation_fun=Tanh2().f_tanh
    ):

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
        self.opt = opt
        self.eps = 1e-10

        # regularizer
        self.reg = L2_norm
        self.lam = 0.0  # 1e-8

        # learning
        self.count = 0
        self.eta = 3e-4  # 1e-4

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
