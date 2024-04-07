from ..Models.dqn_policy import DQNPolicy
from ..Layers.LinearLayer import LinearLayer
from ..utils import (
    initialize_parameters,
)  # import function to initialize weights and biases
from ..Storage.model import ModelDB
from ..wrappers import get_activation_layer_function
import copy
import numpy as np
import random

from ..ActivationFunctions.ActivationLayers import Iden


def argmax_rand(dict_arr):
    """Return key with maximum value, break ties randomly."""
    assert isinstance(dict_arr, dict)
    # Find the maximum value in the dictionary
    max_value = max(dict_arr.values())
    # Get a list of keys with the maximum value
    max_keys = [key for key, value in dict_arr.items() if value == max_value]
    # Randomly select one key from the list
    selected_key = random.choice(max_keys)
    # Return the selected key
    return selected_key


class DQNAgent:
    def __init__(
        self,
        actions,
        input_dim,
        hidden_dim,
        int_type="xavier",
        opt="adam",
        model_db=None,
        lr=0.005,
        eps=0.1,
        gamma=0.99,
        out_type="sig",
        last_line=False,
    ):
        self.actions = actions
        out_dim = len(actions)
        self.model = DQNPolicy(lr=lr)
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
        self.eps = eps
        self.GAMMA = gamma
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

    def act(self, x, model_id, allowed=None, not_allowed=None):
        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()

        if allowed is None:
            valid_actions = self.actions
        else:
            valid_actions = allowed
        if not_allowed is not None:
            valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)

        if random.random() < self.eps:

            action = random.choice(valid_actions)
        else:
            action_probs = self.model.predict(x=x, model=model)
            action_probs_dict = {a: action_probs[a] for a in valid_actions}
            action = argmax_rand(action_probs_dict)

        return action

    def learn(
        self,
        state,
        next_state,
        action,
        reward,
        model_id,
        done=False,
        loss_function="MSE",
        print_cost=False,
    ):
        if isinstance(state, list):
            state = np.array(state).reshape(-1, 1)
        if isinstance(next_state, list):
            next_state = np.array(next_state).reshape(-1, 1)
        if isinstance(reward, (float, int)):
            reward = np.array([reward]).reshape(-1, 1)

        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()

        Q_value = self.model.predict(x=state, model=model)
        t = self.model.predict(next_state, model=model)
        a = np.argmax(t, axis=0)
        Q_value[int(action)] = reward + np.logical_not(done) * self.GAMMA * t[a]
        model = self.model.fit(
            state,
            Q_value,
            model,
            loss_function=loss_function,
            print_cost=print_cost,
        )
        self._model_storage.save_model(model=model, model_id=model_id, w_type="model")

    def _get_valid_actions(self, forbidden_actions, all_actions=None):
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if all_actions is None:
            all_actions = self.actions
        if forbidden_actions is None:
            forbidden_actions = set()
        else:
            forbidden_actions = set(forbidden_actions)

        if not all(a in all_actions for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(all_actions) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError(
                "All actions are forbidden. You must allow at least 1 action."
            )

        valid_actions = list(valid_actions)
        return valid_actions
