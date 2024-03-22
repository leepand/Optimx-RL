from ..LossFunctions.Qef import Qef

### models
class DQNModel:
    """
    multi layer perceptron model:
        mlp(List_with_layers)

    parameters:
        Layerlist   : list of layers
        erf         : errorfunction
        loss        : last training loss

    """

    def __init__(self, Layerlist, no_bias=True):
        self.Layerlist = Layerlist
        self.erf = Qef().Loss  # qef
        self.no_bias = no_bias

    def infer(self, input_, w_s, predict=False):
        """
        compute full forward pass
        """
        out = input_
        i = 0
        x_list = []
        h_list = []
        for L in self.Layerlist:
            out_, h, x1 = L.forward(out, w_s[f"w{i}"])
            x_list.append(x1)
            h_list.append(h)
            out = out_.T
            i += 1
        if predict:
            return out
        else:
            return out, h_list, x_list

    def train(self, Q_, input_, target_, h1, model, x):
        """
        training step
        """
        # outs, h_list = self.infer(input_, w_s=model)
        Q_, h1, input_ = self.infer(x, w_s=model, predict=False)
        outs = Q_
        self.loss = self.erf(target_, outs)
        grad = self.erf(target_, outs, True)
        i = 0
        weights = {}
        x_list = input_
        h_list = h1
        layers = len(self.Layerlist)
        for L in self.Layerlist[::-1]:
            # print(grad.shape, 'before grad')
            layers -= 1
            h = h_list[layers]
            x1 = x_list[layers]
            w = model[f"w{layers}"]
            grad, delta_W = L.backward(grad, x1=x1, h1=h, w=w)
            # print(grad.shape, 'after grad')
            m1 = model[f"m1{layers}"]
            m2 = model[f"m2{layers}"]
            w, m1, m2 = L.update(m1=m1, m2=m2, w=w, delta_W=delta_W)
            model[f"w{layers}"] = w
            weights[f"w{layers}"] = w
            model[f"m1{layers}"] = m1
            model[f"m2{layers}"] = m2
            i += 1
        # update model update times
        weights["model_updated_cnt"] = model["model_updated_cnt"]
        return model, weights
