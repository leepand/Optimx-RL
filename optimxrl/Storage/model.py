"""
Model storage
"""
from abc import abstractmethod
import pickle


class ModelStorage(object):
    """The object to store the model."""

    @abstractmethod
    def get_model(self):
        """Get model"""
        pass

    @abstractmethod
    def save_model(self):
        """Save model"""
        pass


class MemoryModelStorage(ModelStorage):
    """Store the model in memory."""

    def __init__(self):
        self._model = None

    def get_model(self):
        return self._model

    def save_model(self, model):
        self._model = model


class ModelDB(ModelStorage):
    def __init__(self, model_db):
        self._model_db = model_db

    def get_model(self, model_id, w_type="model"):
        VALID_W_TYPES = ("model", "params")
        assert (
            w_type in VALID_W_TYPES
        ), f"Model weights type {w_type} not supported. Valid weights types are {VALID_W_TYPES}"
        model_key = f"{model_id}:{w_type}"
        model = self._model_db.get(model_key)
        if model is not None:
            model = pickle.loads(model)
        return model

    def save_model(self, model, model_id, w_type="model"):
        VALID_W_TYPES = ("model", "params")
        assert (
            w_type in VALID_W_TYPES
        ), f"Model weights type {w_type} not supported. Valid weights types are {VALID_W_TYPES}"
        model_key = f"{model_id}:{w_type}"
        model["model_updated_cnt"] += 1
        self._model_db.set(model_key, pickle.dumps(model))
