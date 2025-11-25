import joblib
import mlflow.pyfunc

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        return cls(model)
