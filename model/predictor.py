from model.connections import Transformer
from model.utils import Iterator


class Predictor(Transformer):
	"""A transformer than takes input data and produces predicted output based on
	the prediction model that it has

	Input:
		x: input data, either as an Iterator, as a batch, or entire data set
		batch_size: required only if x is the entire data set, if batch_size is None, x is treated
					as a batch itself
		y: expected output, used for evaluations

	Output:
		predictions: predictions produced for the input data (x) based on appropriate Keras methods

	Args:
		predictor: a ``Model`` instance that is used to do the predictions
	"""
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
            z = self.predictor.model.predict(x, batch_size, verbose=1)
        else:
            z = self.predictor.model.predict_on_batch(x)
        return {'predictions': z}


class Evaluator(Predictor):
	""" A predictor-like transformer, but instead of producing predictions, it 
	produces evaluation of the model on the input data and expected output data.

	Output:
		evaluation: evaluations for the model based on input data
	"""
    __out__ = ('evaluation',)

    def __transform__(self, x, y=None, batch_size=None):
        if x is Iterator:
            z = self.predictor.model.evaluate_generator(x)
        elif batch_size is not None:
            z = self.predictor.model.evaluate(x, y, batch_size)
        else:
            z = self.predictor.model.test_on_batch(x, y)
        return {'evaluation': z}