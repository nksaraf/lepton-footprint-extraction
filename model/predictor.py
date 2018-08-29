from model.connections import Transformer
from model.utils import Iterator


class Predictor(Transformer):
    __out__ = ('predictions', )

    def __init__(self, name, predictor, need_setup=True):
        super(Predictor, self).__init__(name, need_setup=need_setup)
        self.predictor = predictor

    def __setup__(self, path, load_weights):
        self.predictor.setup(path=path, load_weights=load_weights)

    def __transform__(self, x, y=None, batch_size=None):
        if x is Iterator:
            z = self.predictor.model.predict_generator(x)
        elif batch_size is not None:
            z = self.predictor.model.predict(x, batch_size)
        else:
            z = self.predictor.model.predict_on_batch(x)
        return {'predictions': z}


class Evaluator(Predictor):
    __out__ = ('evaluation',)

    def __transform__(self, x, y=None, batch_size=None):
        if x is Iterator:
            z = self.predictor.model.evaluate_generator(x)
        elif batch_size is not None:
            z = self.predictor.model.evaluate(x, y, batch_size)
        else:
            z = self.predictor.model.test_on_batch(x, y)
        return {'evaluation': z}