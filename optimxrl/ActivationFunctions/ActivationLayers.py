import numpy as np  # import numpy library


class SigmoidLayer:
    """
    This file implements activation layers
    inline with a computational graph model

    Args:
        shape: shape of input to the layer

    Methods:
        forward(Z)
        backward(upstream_grad)

    """

    def __init__(self, output_dim):
        """
        The consturctor of the sigmoid/logistic activation layer takes in the following arguments

        Args:
            shape: shape of input to the layer
        """
        self.layer_type = "Sigmoid"
        self.units = output_dim
        # self.A = np.zeros(shape)  # create space for the resultant activations

    def __str__(self):
        return f"{self.layer_type} Layer"

    def forward(self, Z, predict=True):
        """
        This function performs the forwards propagation step through the activation function

        Args:
            Z: input from previous (linear) layer
        """
        A = 1 / (1 + np.exp(-Z))  # compute activations
        if predict:
            return A
        else:
            return Z

    def backward(self, upstream_grad, A):
        """
        This function performs the  back propagation step through the activation function
        Local gradient => derivative of sigmoid => A*(1-A)

        Args:
            upstream_grad: gradient coming into this layer from the layer above

        """
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        dZ = upstream_grad * A * (1 - A)
        return dZ


class ReLU:
    """ReLU.

    Rectified linear unit layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layer.

    """

    def __init__(self, output_dim):
        self.units = output_dim
        self.layer_type = "ReLU"

    def _len_(self):
        return self.units

    def __str__(self):
        return f"{self.layer_type} Layer"

    def forward(self, Z, predict=True):
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.

        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.

        """
        A = np.maximum(0, Z)
        if predict:
            return A
        else:
            return Z

    def backward(self, upstream_grad, A):
        # np.maximum(0, np.sign(a))
        return upstream_grad * np.heaviside(A, 0.0)


class Tanh:
    """Tanh.

    Hyperbolic tangent layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layers.

    References
    ----------
    [1] Wolfram Alpha - Hyperbolic tangent:
    http://mathworld.wolfram.com/HyperbolicTangent.html

    [2] Brendan O'Connor - tanh is a rescaled logistic sigmoid function:
    https://brenocon.com/blog/2013/10/tanh-is-a-rescaled-logistic-sigmoid-
    function/

    """

    def __init__(self, output_dim):
        self.units = output_dim
        self.layer_type = "Tanh"

    def _len_(self):
        return self.units

    def __str__(self):
        return f"{self.layer_type} Layer"

    def forward(self, Z, predict=True):
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        Z : numpy.Array
            Forward propagation of the previous layer.

        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.

        """
        A = np.tanh(Z)
        if predict:
            return A
        else:
            return Z

    def backward(self, upstream_grad, A):
        """Backward.

        Computes backward propagation pass of this layer.

        Returns
        -------
        dJ : numpy.Array
            Gradient of this layer.

        """
        return upstream_grad * (1 - np.square(A))


class Swish:
    """Swish.

    Swish layer. Swish is a self-gated activation function discovered by
    researchers at Google.

    References
    ----------
    [1] Prajit Ramachandran, Barret Zoph, Quoc V. Le - Swish: A self-gated
    activation function:
    https://arxiv.org/pdf/1710.05941v1.pdf

    """

    def __init__(self, output_dim):
        self.units = output_dim
        self.layer_type = "Swish"

    def _len_(self):
        return self.units

    def __str__(self):
        return f"{self.layer_type} Layer"

    def forward(self, Z, predict=True):
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        Z : numpy.Array
            Forward propagation of the previous layer.

        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.

        """
        A = (1 / (1 + np.exp(-Z))) * Z
        if predict:
            return A
        else:
            return Z

    def backward(self, upstream_grad, A):
        """Backward.

        Computes backward propagation pass of this layer.

        Returns
        -------
        dJ/upstream_grad : numpy.Array
            Gradient of this layer.

        """
        swish = A
        sig = 1 / (1 + np.exp(-swish))

        return upstream_grad * (1 * sig + swish * (1 - swish))


class LeakyReLU:
    def __init__(self, output_dim, alpha=0.2):
        self.units = output_dim
        self.alpha = alpha
        self.layer_type = "leaky"

    def leakyrelu(self, x):
        # return np.where(x >= 0, x, self.alpha * x)
        return np.heaviside(x, self.alpha * x)

    def leakyrelu_derivative(self, x):
        # return np.where(x >= 0, 1, self.alpha)
        return np.heaviside(x, self.alpha)

    def forward(self, Z, predict=True):
        A = self.leakyrelu(Z)
        if predict:
            return A
        else:
            return Z

    def backward(self, upstream_grad, A):
        # return # (self.leakyrelu_derivative(A) * upstream_grad.T).T
        return upstream_grad * self.leakyrelu_derivative(A)


class Iden:
    def __init__(self, output_dim):
        self.units = output_dim
        self.layer_type = "iden"

    def forward(self, Z, predict=True):
        A = Z
        if predict:
            return A
        else:
            return Z

    def backward(self, upstream_grad, A):
        # return # (self.leakyrelu_derivative(A) * upstream_grad.T).T
        return upstream_grad * np.ones(A.shape)
