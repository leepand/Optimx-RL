import optimxrl.Layers.MLPLayer as nn
from ..Models.reward_model import RewardModel
from ..LossFunctions import Qef
from ..ActivationFunctions.Tanh import Tanh2
from ..Storage.model import ModelDB


class RewardAgent:
    def __init__(
        self,
        actions=[],
        input_dim=3,
        hidden_dim=32,
        model_db=None,
        no_bias=False,
        trainer_config=None,
        activation_fun=Tanh2().f_tanh,
    ):
        self._model_db = model_db
        self.actions = actions
        self.epsilon_decay = 0.001
        # STEP 1: create configuration
        self.config = trainer_config(
            model_name="model-v0", actions=actions, input_dim=input_dim
        )
        self.config.OUTPUT_SIZE = 1
        self.no_bias = no_bias
        # STEP 1: create configuration
        A1 = nn.layer(self.config.INPUT_SIZE, hidden_dim, no_bias, activation_fun)
        # A2 = nn.layer(128, 64)
        AOUT = nn.layer(hidden_dim, self.config.OUTPUT_SIZE, no_bias, activation_fun)
        AOUT.f = nn.f_iden
        L1 = nn.layer(self.config.INPUT_SIZE, hidden_dim, no_bias, activation_fun)
        # L2 = nn.layer(128, 64)
        LOUT = nn.layer(hidden_dim, self.config.OUTPUT_SIZE, no_bias, activation_fun)
        LOUT.f = nn.f_iden
        self.onlineNet = RewardModel([A1, AOUT], no_bias=self.no_bias)
        self.targetNet = RewardModel([L1, LOUT], no_bias=self.no_bias)

        self.hidden_dim = hidden_dim
        self.eps = self.config.epsilon
        self._model_storage = ModelDB(model_db=model_db)

    def _init_model(self):
        params = {}
        Layerlist = []
        i = 0
        l0 = nn.layer(self.config.INPUT_SIZE, self.hidden_dim, self.no_bias)
        l1 = nn.layer(self.hidden_dim, self.config.OUTPUT_SIZE, self.no_bias)
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
        l0 = nn.layer(self.config.INPUT_SIZE, self.hidden_dim, self.no_bias)
        l1 = nn.layer(self.hidden_dim, self.config.OUTPUT_SIZE, self.no_bias)
        Layerlist = [l0, l1]
        for L in Layerlist:
            weights[f"w{i}"] = L.w
            i += 1
        weights["model_updated_cnt"] = 0
        return weights

    def predict(self, x, model_id):
        model_weights = self._model_storage.get_model(
            model_id=model_id, w_type="params"
        )
        if model_weights is None:
            model_weights = self._init_weights()
        action_probs = self.onlineNet.infer(input_=x, w_s=model_weights, predict=True)
        return action_probs

    def learn(self, x, y, model_id):
        """updates the onlineDQN with target Q values for
        the greedy action(choosen by onlineDQN)
        """
        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()
        y_hat, h_list, x_list = self.onlineNet.infer(x, w_s=model, predict=False)

        model, weights = self.onlineNet.train(
            y=y_hat, input_=x_list, target_=y, h1=h_list, model=model
        )
        self._model_storage.save_model(model=model, model_id=model_id, w_type="model")
        self._model_storage.save_model(
            model=weights, model_id=model_id, w_type="params"
        )
