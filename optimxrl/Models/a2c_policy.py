from ..LossFunctions.cost_functions import (
    compute_bce_cost,
    compute_stable_bce_cost,
    compute_mse_cost,
)


class A2CPolicy:
    """Model.

    A class to create a model. The model currently supports only sequential
    connections. Networks that require skip-connections cannot be built.

    Example
    -------
    >>> model = Model()
    >>> model.add(Linear(2,5))
    >>> model.add(ReLU(5))
    >>> model.add(Linear(5,2))
    >>> model.add(Sigmoid(2))
    >>> model.train(X, Y, 0.05, 400, "BinaryCrossEntropy, verbose=True")
    >>> model.predict(X)

    """

    def __init__(self, lr=0.1, net_type="actor"):
        self.layers = []
        self.loss = []
        self.lr = lr
        self.net_type = net_type

    def add(self, layer):
        """Adds.

        Adds a new layer to the model.

        Parameters
        ----------
        layer : freya.Layer
            A freya layer to add to the model.
        """
        self.layers.append(layer)

    def predict(self, x, model, update_mode=False):
        """Predicts.

        Given a set of data X, this function makes a prediction of X.

        Parameters
        ----------
        X : numpy.Array
            Data to make the predictions.

        Returns
        -------
        forward : numpy.Array
            Prediction for the given dataset.

        """
        # Forward pass
        x_list = []
        for i, _ in enumerate(self.layers):
            if self.layers[i].layer_type != "Linear":
                forward = self.layers[i].forward(x)
                x_list.append(forward)
            else:
                W = model[f"{self.net_type}_W{i}"]
                b = model[f"{self.net_type}_b{i}"]
                A_prev, forward = self.layers[i].forward(x, W, b, False)
                x_list.append(A_prev)
            x = forward
        if update_mode:
            return forward, x_list
        else:
            return forward

    def fit(self, x_list, y_hat, y, model, loss_function="BE", print_cost=False):
        """Runs epoch.

        Helper function of train procedure.

        Parameters
        ----------
        X_train : numpy.Array
            Training data. Must match the input size of the first layer.
        Y_train : numpy.Array
            Training labels.
        learning_rate : float
            Number of epochs to train the model
        epochs : int
            asdad
        loss_function : str
            Chosen function to compute loss.

        Returns
        -------
        error : float
            Model error in this epoch.

        """
        # Forward pass

        forward = y_hat
        # Compute loss and first gradient
        if loss_function == "BE":
            cost, dY_hat = compute_stable_bce_cost(y, forward)
        elif loss_function == "MSE":
            cost, dY_hat = compute_mse_cost(y, forward)
        else:
            raise ValueError(f"{loss_function} is not supported.")
        if print_cost:
            print("Cost at epoch#{}: {}".format("yhat", cost))
        # $ print(x_list,"x_list")
        # Backpropagation
        layers = len(self.layers)
        # for i, _ in reversed(list(enumerate(self.layers))):
        for L in self.layers[::-1]:
            layers -= 1
            xi = x_list[layers]
            if L.layer_type != "Linear":
                dY_hat = L.backward(dY_hat, xi)
                # print(dY_hat, xi, "noline")
            else:
                # print(dY_hat, xi, "line")
                if L.opt in ["rmsprop", "adam"]:
                    W = model[f"{self.net_type}_W{layers}"]
                    b = model[f"{self.net_type}_b{layers}"]
                    m1 = model[f"{self.net_type}_m1{layers}"]
                    m2 = model[f"{self.net_type}_m2{layers}"]
                    model_updated_cnt = model["model_updated_cnt"]
                    dY_hat, dW, dB = L.backward(dY_hat, xi, W)
                    W, b, m1, m2 = L.update_params(
                        dW,
                        dB,
                        W,
                        b,
                        m1=m1,
                        m2=m2,
                        model_updated_cnt=model_updated_cnt,
                        lr=self.lr,
                    )
                    model[f"{self.net_type}_W{layers}"] = W
                    model[f"{self.net_type}_b{layers}"] = b
                    model[f"{self.net_type}_m1{layers}"] = m1
                    model[f"{self.net_type}_m2{layers}"] = m2
                else:
                    W = model[f"{self.net_type}_W{layers}"]
                    b = model[f"{self.net_type}_b{layers}"]
                    # print(dY_hat.shape, xi.shape, "before")
                    dY_hat, dW, dB = L.backward(dY_hat, xi, W)
                    # print(dY_hat.shape, dW.shape, "after")
                    W, b = L.update_params(
                        dW, dB, W, b, lr=self.lr
                    )  # optimize(dW, dB, learning_rate)

                    model[f"{self.net_type}_W{layers}"] = W
                    model[f"{self.net_type}_b{layers}"] = b
        return model
