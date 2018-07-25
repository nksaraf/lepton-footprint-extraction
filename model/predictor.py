from model.connections import Transformer
from model.utils import Iterator


class Predictor(Transformer):
    __out__ = ('output', )

    def __init__(self, name, predictor, mode):
        super(Predictor, self).__init__(name, need_setup=True)
        self.predictor = predictor
        self.mode = mode

    def __setup__(self, model_path):
        self.predictor.setup(model_path=model_path)

    def __transform__(self, x, y=None, batch_size=None):
        if self.mode == 'predict':
            if x is Iterator:
                z = self.predictor.model.predict_generator(x)
            elif batch_size is not None:
                z = self.predictor.model.predict(x, batch_size)
            else:
                z = self.predictor.model.predict_on_batch(x)
        else:
            if x is Iterator:
                z = self.predictor.model.evaluate_generator(x)
            elif batch_size is not None:
                z = self.predictor.model.evaluate(x, y, batch_size)
            else:
                z = self.predictor.model.test_on_batch(x, y)

        return {'output': z}