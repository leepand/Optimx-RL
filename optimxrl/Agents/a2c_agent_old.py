from ..Models.a2c_policy import A2CPolicy
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


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x))


class A2CAgent:
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
        actor_out_type="relu",
        critic_out_type="relu",
        actor_last_layer_type="softmax",
        critic_last_layer_type="iden",
    ):
        self.actions = actions
        out_dim = len(actions)
        self.actor_net_type = "actor"
        self.critic_net_type = "critic"
        self.model_actor = self.create_actor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            int_type=int_type,
            opt=opt,
            out_type=actor_out_type,
            last_layer_type=actor_last_layer_type,
            lr=lr,
        )
        self.model_critic = self.create_critic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            int_type=int_type,
            opt=opt,
            out_type=critic_out_type,
            last_layer_type=critic_last_layer_type,
            lr=lr,
        )

        self.opt = opt
        self.eps = eps
        self.GAMMA = gamma
        self.int_type = int_type
        self._model_storage = ModelDB(model_db=model_db)

    def create_actor(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        int_type,
        opt,
        out_type="sig",
        last_layer_type="sig",
        lr=0.005,
    ):
        model_actor = A2CPolicy(lr=lr, net_type=self.actor_net_type)
        # Our network architecture has the shape:
        #       (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid]->[Linear->Sigmoid] -->(output)

        # ------ LAYER-1 ----- define hidden layer that takes in training data
        OutLayer = get_activation_layer_function(out_type=out_type)
        model_actor.add(
            LinearLayer(n_in=input_dim, n_out=hidden_dim[0], ini_type=int_type, opt=opt)
        )
        model_actor.add(OutLayer(hidden_dim[0]))
        # ------ LAYER-2 ----- define output layer that take is values from hidden layer
        if len(hidden_dim) > 1:
            model_actor.add(
                LinearLayer(
                    n_in=hidden_dim[0], n_out=hidden_dim[1], ini_type=int_type, opt=opt
                )
            )
            model_actor.add(OutLayer(hidden_dim[1]))
            # ------ LAYER-3 ----- define output layer that take is values from 2nd hidden layer

            model_actor.add(
                LinearLayer(
                    n_in=hidden_dim[1], n_out=out_dim, ini_type=int_type, opt=opt
                )
            )
        else:
            model_actor.add(
                LinearLayer(
                    n_in=hidden_dim[0], n_out=out_dim, ini_type=int_type, opt=opt
                )
            )
        model_actor.add(OutLayer(last_layer_type))
        return model_actor

    def create_critic(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        int_type,
        opt,
        out_type="sig",
        last_layer_type="sig",
        lr=0.005,
    ):
        model_critic = A2CPolicy(lr=lr, net_type=self.critic_net_type)
        # Our network architecture has the shape:
        #       (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid]->[Linear->Sigmoid] -->(output)

        # ------ LAYER-1 ----- define hidden layer that takes in training data
        OutLayer = get_activation_layer_function(out_type=out_type)
        model_critic.add(
            LinearLayer(n_in=input_dim, n_out=hidden_dim[0], ini_type=int_type, opt=opt)
        )
        model_critic.add(OutLayer(hidden_dim[0]))
        # ------ LAYER-2 ----- define output layer that take is values from hidden layer
        if len(hidden_dim) > 1:
            model_critic.add(
                LinearLayer(
                    n_in=hidden_dim[0], n_out=hidden_dim[1], ini_type=int_type, opt=opt
                )
            )
            model_critic.add(OutLayer(hidden_dim[1]))
            # ------ LAYER-3 ----- define output layer that take is values from 2nd hidden layer

            model_critic.add(
                LinearLayer(
                    n_in=hidden_dim[1], n_out=out_dim, ini_type=int_type, opt=opt
                )
            )
        else:
            model_critic.add(
                LinearLayer(
                    n_in=hidden_dim[0], n_out=out_dim, ini_type=int_type, opt=opt
                )
            )

        model_critic.add(OutLayer(last_layer_type))
        return model_critic

    def _init_model(self):
        params = {}
        Layerlist_actor = copy.deepcopy(self.model_actor.layers)
        i = 0
        net_type = "actor"
        for L in Layerlist_actor:
            if L.layer_type == "Linear":
                _params = initialize_parameters(
                    L.n_in, L.n_out, ini_type=self.int_type, opt=self.opt
                )
                params[f"{net_type}_W{i}"] = _params["W"]
                params[f"{net_type}_b{i}"] = _params["b"]
                if self.opt in ["rmsprop", "adam"]:
                    params[f"{net_type}_m1{i}"] = _params["m1"]
                    params[f"{net_type}_m2{i}"] = _params["m2"]
            i += 1
        Layerlist_critic = copy.deepcopy(self.model_critic.layers)
        net_type = "critic"
        j = 0
        for LC in Layerlist_critic:
            if LC.layer_type == "Linear":
                _params = initialize_parameters(
                    LC.n_in, LC.n_out, ini_type=self.int_type, opt=self.opt
                )
                params[f"{net_type}_W{j}"] = _params["W"]
                params[f"{net_type}_b{j}"] = _params["b"]
                if self.opt in ["rmsprop", "adam"]:
                    params[f"{net_type}_m1{j}"] = _params["m1"]
                    params[f"{net_type}_m2{j}"] = _params["m2"]
            j += 1

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
            # print(action,"r")
        else:
            action_probs = self.model_actor.predict(x=x, model=model)
            # print(action_probs, "action_probs")
            action_probs_dict = {a: action_probs[a] for a in valid_actions}
            action = argmax_rand(action_probs_dict)
            # print(action,"m",action_probs_dict,action_probs)
        return action

    def learn(
        self,
        state,
        next_state,
        action,
        reward,
        model_id,
        done=False,
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

        _action_probs, x_list_actor = self.model_actor.predict(
            x=state, model=model, update_mode=True
        )
        value_curr, x_list_critic = self.model_critic.predict(
            x=state, model=model, update_mode=True
        )
        value_next = self.model_critic.predict(next_state, model=model)
        Q = np.copy(value_curr)
        # print(action,type(action),a,type(a))
        TD_target = reward + (1 - np.logical_not(done)) * self.GAMMA * value_next
        # Q_value[int(action)] = reward + np.logical_not(done) * self.GAMMA * t[a]
        # critic_target= advantage
        advantage = TD_target - value_curr
        action_probs = _action_probs
        ps = np.maximum(1.0e-5, np.minimum(1.0 - 1e-5, action_probs))
        log_probs = np.log(ps)
        log_probs_curr = np.copy(log_probs)
        # print(log_probs,advantage,"advantage")
        log_probs[int(action)] *= advantage[0]
        # dy = advantage[0] / action_probs[int(action)]
        model = self.model_critic.fit(
            x_list_critic,
            value_curr,
            TD_target,
            model,
            loss_function="MSE",
            print_cost=print_cost,
        )
        model = self.model_actor.fit(
            x_list_actor,
            log_probs_curr,
            log_probs,
            model,
            loss_function="BE",
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
