import numpy as np
from optimxrl.Storage.model import ModelDB
import random


class CmabAgent:
    # 不能新增action
    def __init__(self, n_actions, n_features, alpha=0.1, model_db=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self._model_db = model_db
        self._model_storage = ModelDB(model_db=model_db)

        # Initialize parameters
        self.A = np.array(
            [np.identity(n_features) for _ in range(n_actions)]
        )  # action covariance matrix
        self.b = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )  # action reward vector
        self.theta = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )  # action parameter vector

    def act(self, context, model_id, not_allowed=None):
        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            A = self.A
            b = self.b
        else:
            A = model["A"]
            b = model["b"]
        context = np.array(context)  # Convert list to ndarray
        context = context.reshape(
            -1, 1
        )  # reshape the context to a single-column matrix
        valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)
        # p = np.zeros(self.n_actions)
        action_probs_dict = {}
        for a in valid_actions:
            theta = np.dot(np.linalg.inv(A[a]), b[a])  # theta_a = A_a^-1 * b_a
            theta = theta.reshape(-1, 1)  # Explicitly reshape theta
            action_probs_dict[a] = np.dot(theta.T, context) + self.alpha * np.sqrt(
                np.dot(context.T, np.dot(np.linalg.inv(A[a]), context))
            )  # p_t(a|x_t) = theta_a^T * x_t + alpha * sqrt(x_t^T * A_a^-1 * x_t)

        # for a in range(self.n_actions):
        #    theta = np.dot(
        #        np.linalg.inv(self.A[a]), self.b[a]
        #    )  # theta_a = A_a^-1 * b_a
        #    theta = theta.reshape(-1, 1)  # Explicitly reshape theta
        #    p[a] = np.dot(theta.T, context) + self.alpha * np.sqrt(
        #        np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context))
        #    )  # p_t(a|x_t) = theta_a^T * x_t + alpha * sqrt(x_t^T * A_a^-1 * x_t)
        action = self.argmax_rand(action_probs_dict)
        return action

    def learn(self, action, context, reward, model_id):
        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = {}
            model["A"] = np.array(
                [np.identity(self.n_features) for _ in range(self.n_actions)]
            )  # action covariance matrix
            model["b"] = np.array(
                [np.zeros(self.n_features) for _ in range(self.n_actions)]
            )
            # model["theta"] = np.array(
            #    [np.zeros(self.n_features) for _ in range(self.n_actions)]
            # )  # action parameter vector
        model["A"][action] += np.outer(context, context)  # A_a = A_a + x_t * x_t^T
        model["b"][action] += reward * context  # b_a = b_a + r_t * x_tx
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
            all_actions = [a for a in range(self.n_actions)]
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
