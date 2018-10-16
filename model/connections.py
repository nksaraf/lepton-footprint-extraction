from __future__ import print_function
import inspect
import logging
import sys
import types
from abc import ABCMeta, abstractmethod, abstractproperty

from attrdict import AttrDict

"""A tiny pipeline library written by me to help in the prototyping phase as well as 
writing intuitive code that matches the context of a "automation pipeline". Can be used
for any kind of project where there are sequences of tasks to be done.

The analogy is made to a circuit:
	Wire: Represents data flowing between components. The data is represented as a dictionary
		to be used as keyword arguments to functions easily. They are also backed by an AttrDict,
		which allows the dictionary entries to be accessed as attributes (asthetically better).
		Created and used just like a normal dictionary.

	Transformer: Wires are connected to transformers that do some operation on them and return
		another wire as output. The transformer expects certain input, and the wire is resolved to
		match those inputs before connection. This happens internally.

	Though the Transfomer class will help write modular code and will be required if the transformers
	require certain setup to be done and hold attributes themselves before the transforming, they can
	be replaced by reqular functions or even lambda functions. 

	Some syntactic sugar is also introduced here and will become clearer as you read the code:
		(Wire -> x, y | Transformer -> A, B)
		x + y => merged wires
		x(blah=blah ...) => adapting (explained below)
		x | A => connecting x to A
		x | A | B => connecting x to A, the output of this is connected to B
		x > A => connecting x to A
		A(x) => connecting x to A
		(x + y) | A => merged (x,y) connected to A
		x | (A + B) => connecting x to MegaTransfomer A, B

"""

class Wire(AttrDict):
    def __repr__(self):
        contents = ', '.join(['{}: {}'.format(k, v) for k, v in self.iteritems()])
        return 'Wire({contents})'.format(contents=contents)

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self.__adapt__(*args, **kwargs)

    def __adapt__(self, *args, **kwargs):
    	"""Changes the naming of the items in the wire, by a given specification.
    	Used to match the wire to the appropriate inputs required by a Transformer.
		Arguments provided are used to form a dictionary which is used as follows: 
			for every key, value pair in the adapter dictionary:
				if value is an item in the wire, 
					adapted_wire[key] = this[value]
				if the value is not in the wire, then an exception is raised
			all the other keys in the wire are copied to the adapted wire
		"""

        output = Wire()
        adapter = dict(*args, **kwargs)
        for key, value in adapter.iteritems():
            if value in self:
                output[key] = self[value]
            else:
                raise MissingWireError

        for key in self.keys():
            if key not in adapter.itervalues():
                output[key] = self[key]
        return output

    def __dir__(self):
        return dir(dict) + list(self.keys())

    def __or__(self, other):
        return _connect(other, self)

    def __gt__(self, other):
        return self.__or__(other)


class Transformer(object):
	"""Abstract class. Implementing classes need to override __transform__, and 
	__setup__ if necessary. If need_setup=True, the setup function must be classed before
	transform, otherwise there will be an error.
	"""
    __metaclass__ = ABCMeta

    def __init__(self, name='transformer', logger=None, need_setup=False):
        self.name = name.upper()
        if logger is None:
            self.logger = print
        else:
            self.logger = logger
        self.is_setup = not need_setup

    @property
    def inputs(self):
    	"""List of argument names for __transform__ (except self)"""
        return inspect.getargspec(self.__transform__)[0][1:]

    @property
    def outputs(self):
        return self.__class__.__out__

    def setup(self, **kwargs):
    	"""Must be called if need_setup is True"""
        self.is_setup = True
        self.logger('Setting up {}'.format(self.__str__()))
        self.__setup__(**kwargs)
        return self

    def transform(self, **kwargs):
        if not self.is_setup:
            raise TransformerNotSetupError
        else:
            self.logger('Connecting {}'.format(self.__str__()))
            return self.__transform__(**kwargs)

    def connect(self, wire):
        _connect(self, wire)

    def log(self, msg):
        pre = (' ' * len('Connecting ')) + ('{}: '.format(self.name))
        self.logger(pre + msg)

    @abstractproperty
    def __out__(self):
        return tuple()

    def __call__(self, wire):
        return _connect(self, wire)

    def __repr__(self):
        return "{}: ({}) ==> {}{} ==> ({})".format(
            self.name,
            ', '.join(self.inputs),
            self.__class__.__name__,
            '' if self.is_setup else '(needs setup)',
            ', '.join(self.outputs))

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if isinstance(other, Transformer):
            return MegaTransformer(self, other, self.name + '+' + other.name)

    def __setup__(self, **kwargs):
        pass

    @abstractmethod
    def __transform__(self, **kwargs):
        raise NotImplementedError


class StreamTransformer(Transformer):
    __metaclass__ = ABCMeta

    def stream(self, wire):
        pass
        # TODO: what to do


class MegaTransformer(Transformer):
    __out__ = ('outputs', )

    def __init__(self, left, right, name):
        super(MegaTransformer, self).__init__(name, left.logger)
        self.left = left
        self.right = right

    @property
    def inputs(self):
        return self.left.inputs

    def __transform__(self, **kwargs):
        output = self.left.transform(**kwargs)
        output.update(self.right.transform(**kwargs))
        return output


class LambdaTransformer(Transformer):
    __out__ = ('output', )

    def __init__(self, func, name):
        super(LambdaTransformer, self).__init__(name=name)
        self.func = func

    @property
    def inputs(self):
        return inspect.getargspec(self.func)[0]

    def __transform__(self, **kwargs):
        return { self.name.lower() + '_output': self.func(**kwargs) }


def _connect(transformer, wire):
	"""Connects a wire to a transfomer. Checks for all the input arguments required in the wire,
	and adapts the wire for the transformer. then connects and returns the transformed 
	wire.

	Args:
		transformer: `Transformer` or function or lambda
		wire: `Wire` or dictionary

	Returns:
		transformed `Wire`
	"""
    if isinstance(transformer, Transformer):
        inputs = transformer.inputs
        func = transformer.transform
    elif isinstance(transformer, types.FunctionType):
        inputs = inspect.getargspec(transformer)[0]
        func = transformer
    else:
        raise InvalidAdapterError

    plug = {}
    for key in inputs:
        if key in wire:
            plug[key] = wire[key]
    return Wire(func(**plug))


class InvalidAdapterError(BaseException):
    pass


class MissingWireError(BaseException):
    pass


class TransformerNotSetupError(BaseException):
    pass


LOGGER = 'log'


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s', datefmt='%H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt=message_format)
    logger.addHandler(console_handler)

    return logger


def get_logger():
    return logging.getLogger(LOGGER)


init_logger(LOGGER)
