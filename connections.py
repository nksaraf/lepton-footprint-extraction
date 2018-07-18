from attrdict import AttrDict
import inspect
import types
import logging
import sys


class Wire(AttrDict):
    def __repr__(self):
        contents = ', '.join(['{}: {}'.format(k, v) for k, v in self.iteritems()])
        return 'Wire({contents})'.format(contents=contents)

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self.__adapt__(*args, **kwargs)

    def __adapt__(self, *args, **kwargs):
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
    __out__ = tuple()

    def __init__(self, name='transformer', logger=None, need_setup=False):
        self.name = name.upper()
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
        self.is_setup = not need_setup

    def _params(self):
        return {}

    def __inputs__(self):
        return inspect.getargspec(self.__transform__)[0][1:]

    def __outputs__(self):
        return self.__class__.__out__

    def __setup__(self, **kwargs):
        pass

    def __transform__(self, **kwargs):
        raise NotImplementedError

    def setup(self, **kwargs):
        self.is_setup = True
        self.__setup__(**kwargs)
        return self

    def transform(self, **kwargs):
        if not self.is_setup:
            raise TransformerNotSetupError
        else:
            self.logger.info('Connecting {}'.format(self.__str__()))
            return self.__transform__(**kwargs)

    def __call__(self, wire):
        return _connect(self, wire)

    def log(self, msg):
        pre = ' ' * len('Connecting ') + ('{}: '.format(self.name))
        self.logger.info(pre + msg)

    def __repr__(self):
        return "{}: ({}) ==> {}{} ==> ({})".format(
            self.name,
            ', '.join(self.__inputs__()),
            self.__class__.__name__,
            '' if self.is_setup else '(needs setup)',
            ', '.join(self.__outputs__()))

    def __str__(self):
        return self.__repr__()


def _connect(transformer, wire):
    if isinstance(transformer, Transformer):
        inputs = transformer.__inputs__()
        func = transformer.transform
        name = transformer.name
    elif isinstance(transformer, types.FunctionType):
        inputs = inspect.getargspec(transformer)[0]
        func = transformer
        name = 'transformer'
    else:
        raise InvalidAdapterError

    plug = {}
    for key in inputs:
        if key not in wire:
            plug[key] = None
            print("\tWarning: {key} not found in input to {name}".format(key=key, name=name))
        else:
            plug[key] = wire[key]
    return Wire(func(**plug))


class InvalidAdapterError(BaseException):
    pass


class MissingWireError(BaseException):
    pass


class TransformerNotSetupError(BaseException):
    pass


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%H:%M:%S')

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(console_handler)

    return logger


def get_logger():
    return logging.getLogger('log')


init_logger('log')


# class Adapter(object):
#     def __init__(self, *args, **kwargs):
#         self.adapter = dict(*args, **kwargs)
#         check = [type(k) == str and type(v) == str for k, v in self.adapter.iteritems()]
#         if not all(check):
#             raise InvalidAdapterError
#
#     def __adapt__(self, wire):
#         output = Wire()
#         for key, value in self.adapter.iteritems():
#             if value in wire:
#                 output[key] = wire[value]
#             else:
#                 raise MissingWireError
#
#         for key in wire.keys():
#             if key not in self.adapter.itervalues():
#                 output[key] = wire[key]
#         return output
#
#     def __repr__(self):
#         contents = ', '.join(['{}: {}'.format(k, v) for k, v in self.adapter.iteritems()])
#         return 'Adapter({contents})'.format(contents=contents)
#
#     def __str__(self):
#         return self.__repr__()
