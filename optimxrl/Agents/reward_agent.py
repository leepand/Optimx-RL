from ..Models.model import Model
from ..Layers.LinearLayer import LinearLayer
from ..utils import (
    initialize_parameters,
)  # import function to initialize weights and biases
from ..Storage.model import ModelDB
from ..wrappers import get_activation_layer_function
import copy
import numpy as np
from ..ActivationFunctions.ActivationLayers import Iden


class RewardAgent:
    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        int_type="xavier",
        opt="adam",
        model_db=None,
        lr=0.5,
        out_type="sig",
        last_line=False,
    ):
        self.model = Model(lr=lr)
        # Our network architecture has the shape:
        #       (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid]->[Linear->Sigmoid] -->(output)

        # ------ LAYER-1 ----- define hidden layer that takes in training data
        OutLayer = get_activation_layer_function(out_type=out_type)
        self.model.add(
            LinearLayer(n_in=input_dim, n_out=hidden_dim[0], ini_type=int_type, opt=opt)
        )
        self.model.add(OutLayer(hidden_dim[0]))
        # ------ LAYER-2 ----- define output layer that take is values from hidden layer
        self.model.add(
            LinearLayer(
                n_in=hidden_dim[0], n_out=hidden_dim[1], ini_type=int_type, opt=opt
            )
        )
        self.model.add(OutLayer(hidden_dim[1]))
        # ------ LAYER-3 ----- define output layer that take is values from 2nd hidden layer

        self.model.add(
            LinearLayer(n_in=hidden_dim[1], n_out=out_dim, ini_type=int_type, opt=opt)
        )
        if last_line:
            self.model.add(Iden(out_dim))
        else:
            self.model.add(OutLayer(out_dim))

        self.opt = opt
        self.int_type = int_type
        self._model_storage = ModelDB(model_db=model_db)

    def _init_model(self):
        params = {}
        Layerlist = copy.deepcopy(self.model.layers)
        i = 0

        for L in Layerlist:
            if L.layer_type == "Linear":
                _params = initialize_parameters(
                    L.n_in, L.n_out, ini_type=self.int_type, opt=self.opt
                )
                params[f"W{i}"] = _params["W"]
                params[f"b{i}"] = _params["b"]
                if self.opt in ["rmsprop", "adam"]:
                    params[f"m1{i}"] = _params["m1"]
                    params[f"m2{i}"] = _params["m2"]
            i += 1
        params["model_updated_cnt"] = 0
        return params

    def predict(self, x, model_id):
        if isinstance(x, list):
            x = np.array([1, 0]).reshape(-1, 1)
        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()
        result = self.model.predict(x=x, model=model)

        return result

    def learn(self, x, y, model_id, loss_function="MSE", print_cost=False):
        if isinstance(x, list):
            x = np.array([1, 0]).reshape(-1, 1)
        if isinstance(y, (float, int)):
            y = np.array([y]).reshape(-1, 1)

        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()

        model = self.model.fit(
            x, y, model, loss_function=loss_function, print_cost=print_cost
        )
        self._model_storage.save_model(model=model, model_id=model_id, w_type="model")
