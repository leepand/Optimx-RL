from optimxrl.Storage.model import ModelDB
import optimxrl.Layers.MLPLayer as nn
from optimxrl.Models.reward_model import RewardModel

import random
import numpy as np


class NeuralBandit:
    def __init__(
        self,
        act_dim,
        feature_dim,
        hidden_dim=32,
        lr=0.01,
        eps=0.2,
        model_db=None,
        no_bias=False,
    ):
        self.actions = [a for a in range(act_dim)]
        self.n_actions = act_dim
        self.n_features = feature_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.eps = eps
        self._model_db = model_db
        self._model_storage = ModelDB(model_db=model_db)

        self.no_bias = no_bias
        # STEP 1: create configuration
        A1 = nn.layer(feature_dim, hidden_dim, no_bias)
        # A2 = nn.layer(128, 64)
        AOUT = nn.layer(hidden_dim, act_dim, no_bias)
        AOUT.f = nn.f_iden
        self.onlineNet = RewardModel([A1, AOUT], no_bias=self.no_bias)

    def _init_model(self):
        params = {}
        Layerlist = []
        i = 0
        l0 = nn.layer(self.n_features, self.hidden_dim, self.no_bias)
        l1 = nn.layer(self.hidden_dim, self.n_actions, self.no_bias)
        Layerlist = [l0, l1]
        for L in Layerlist:
            params[f"w{i}"] = L.w
            params[f"m1{i}"] = L.m1
            params[f"m2{i}"] = L.m2
            # print(L.w,"w",L.w.shape,L.m1,L.m2,"m")
            i += 1
        params["model_updated_cnt"] = 0
        return params

    def _init_weights(self):
        weights = {}
        Layerlist = []
        i = 0
        l0 = nn.layer(self.n_features, self.hidden_dim, self.no_bias)
        l1 = nn.layer(self.hidden_dim, self.n_actions, self.no_bias)
        Layerlist = [l0, l1]
        for L in Layerlist:
            weights[f"w{i}"] = L.w
            i += 1
        weights["model_updated_cnt"] = 0
        return weights

    def act(self, x, model_id, allowed=None, not_allowed=None):
        if isinstance(x, list):
            x = np.array(x)[True, :]
        model_weights = self._model_storage.get_model(
            model_id=model_id, w_type="params"
        )
        if model_weights is None:
            model_weights = self._init_weights()
        if allowed is None:
            valid_actions = self.actions
        else:
            valid_actions = allowed
        if not_allowed is not None:
            valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)

        if random.random() < self.eps:

            action = random.choice(valid_actions)
        else:
            action_probs = self.onlineNet.infer(
                input_=x, w_s=model_weights, predict=True
            )
            action_probs_dict = {
                a: action_probs[range(action_probs.shape[0]), a] for a in valid_actions
            }
            action = self.argmax_rand(action_probs_dict)
        return action

    def learn(self, x, action, reward, model_id):
        if isinstance(x, list):
            x = np.array[True, :]
        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()
        y_hat, h_list, x_list = self.onlineNet.infer(x, w_s=model, predict=False)
        model, weights = self.onlineNet.train(
            y=y_hat, input_=x_list, target_=reward, h1=h_list, model=model
        )
        self._model_storage.save_model(model=model, model_id=model_id, w_type="model")
        self._model_storage.save_model(
            model=weights, model_id=model_id, w_type="params"
        )

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

    def argmax_rand(self, dict_arr):
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
