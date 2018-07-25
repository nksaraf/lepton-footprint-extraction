from model.connections import Transformer
from model.utils import Iterator


class Predictor(Transformer):
    __out__ = ('output', )

    def __init__(self, name, mode):
        super(Predictor, self).__init__(name, need_setup=True)
        self.model = None
        self.mode = mode

    def __setup__(self, model):
        self.model = model

    def __transform__(self, x, y=None, batch_size=None):
        if self.mode == 'predict':
            if x is Iterator:
                z = self.model.predict_generator(x)
            elif batch_size is not None:
                z = self.model.predict(x, batch_size)
            else:
                z = self.model.predict_on_batch(x)
        else:
            if x is Iterator:
                z = self.model.evaluate_generator(x)
            elif batch_size is not None:
                z = self.model.evaluate(x, y, batch_size)
            else:
                z = self.model.test_on_batch(x, y)

        return {'output': z}