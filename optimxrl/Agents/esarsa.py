import random
import numpy as np
import pickle


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


class ExpectedSarsaAgent:
    def __init__(
        self,
        actions,
        epsilon=0.2,
        alpha=0.4,
        gamma=0.99,
        model_db=None,
        score_db=None,
        random_seed=0,
    ):
        """
        Constructor
        Args:
                epsilon: The degree of exploration
                gamma: The discount factor
                num_state: The number of states
                action_space: To call the random action
        """
        self.Q = {}
        self.actions = actions
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma
        random.seed(random_seed)

        self._model_db = model_db
        self._score_db = score_db

    def get_Q_value(self, Q_dict, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return Q_dict.get(
            (state, action), 0.0
        )  # Return 0.0 if state-action pair does not exist

    def act(self, state, model_id=None, not_allowed=None, allowed=None, topN=1):
        if allowed is not None:
            valid_actions = allowed
        else:
            valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)
        # Choose a random action
        explore = np.random.binomial(1, self.eps)
        if explore == 1:
            # action = random.choice(self.actions)

            actions_list = np.random.choice(
                valid_actions, size=topN, replace=False, p=None
            ).tolist()
            action = actions_list[0]
        # Choose the greedy action
        else:
            Q_state_scores = self.get_Q_scores(
                state, model_id, topN=-1, withscores=False
            )
            if Q_state_scores:
                valid_actions_dict = {
                    a: score
                    for a, score in Q_state_scores.items()
                    if a in valid_actions
                }
                if valid_actions_dict:
                    action = argmax_rand(dict_arr=Q_state_scores)
                else:
                    action = random.choice(valid_actions)
            else:
                action = random.choice(valid_actions)

        return action

    def learn(self, state, action, reward, next_state, model_id=None):
        """
        Expected Sarsa update
        """
        _Q_dict = self.get_model(model_id=model_id)

        if _Q_dict is None:
            Q_dict = {}
        else:
            Q_dict = _Q_dict

        # Probability for taking each action in next state
        next_state_probs = self.action_probs(next_state, model_id, 1)
        # Q-values for each action in next state
        q_next_state = [
            self.get_Q_value(Q_dict, next_state, action) for action in self.actions
        ]
        next_state_expectation = sum(
            [a * b for a, b in zip(next_state_probs, q_next_state)]
        )

        # If this is the first time the state action pair is encountered
        q_current = Q_dict.get((state, action), None)
        if q_current is None:
            _Q_score = reward
        else:
            _Q_score = q_current + (
                self.alpha * (reward + self.gamma * next_state_expectation - q_current)
            )
            # print(_Q_score,q_current,next_state_expectation)

        Q_dict[(state, action)] = _Q_score
        score_key = f"{model_id}:{state}"
        old_score = self.get_score(model_id=score_key)
        old_score[action] = _Q_score
        self.save_model(model=Q_dict, model_id=model_id)
        self.save_score(model=old_score, model_id=score_key)

    def action_probs(self, state, model_id, topN=1):
        """
        Returns the probability of taking each action in the next state.
        """
        next_state_probs = [self.eps / len(self.actions)] * len(self.actions)
        best_action_dict = self.greedy_action_selection(state, model_id, topN)
        if not best_action_dict:
            best_action = random.choice(self.actions)
        else:
            best_action = argmax_rand(best_action_dict)
        best_action_index = self.actions.index(best_action)
        # next_state_probs[best_action] += 1.0 - epsilon
        next_state_probs[best_action_index] += 1.0 - self.eps

        return next_state_probs

    def greedy_action_selection(self, state, model_id=None, topN=1):
        """
        Selects action with the highest Q-value for the given state.
        """
        # Get all the Q-values for all possible actions for the state
        maxQ_action_dict = self.get_Q_scores(state, model_id, topN, withscores=False)
        return maxQ_action_dict

    def get_Q_scores(self, state, model_id, topN, withscores=False):
        score_key = f"{model_id}:{state}"
        score_dict = self.get_score(model_id=score_key)
        return score_dict

    def get_random_action(self, topN):
        if topN > len(self.actions):
            raise Exception("topN is longer than len of self.actions")
        action_list = np.random.choice(
            self.actions, size=topN, replace=False, p=None
        ).tolist()
        return action_list

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
            # raise ValueError(
            #    "All actions are forbidden. You must allow at least 1 action."
            # )
            return None

        valid_actions = list(valid_actions)
        return valid_actions

    def get_score_key(self, model_id, state):
        return f"{model_id}:{state}:Qscore"

    def get_model_key(self, model_id):
        return f"{model_id}:qvalue"

    def get_model(self, model_id):
        model_key = f"{model_id}:qvalue"
        model = self._model_db.get(model_key)
        if model is not None:
            model = pickle.loads(model)
        return model

    def save_model(self, model, model_id):
        model_key = f"{model_id}:qvalue"
        if isinstance(model, dict):
            self._model_db.set(model_key, pickle.dumps(model))

    def get_score(self, model_id):
        model_key = f"{model_id}:Qscore"
        model = self._score_db.get(model_key)
        if model is not None:
            model = pickle.loads(model)
        else:
            model = {}
        return model

    def save_score(self, model, model_id):
        model_key = f"{model_id}:Qscore"
        if isinstance(model, dict):
            self._score_db.set(model_key, pickle.dumps(model))
