from abc import ABCMeta
from model.connections import Transformer
from model.utils import ModelCheckpoint


class Trainer(Transformer):
	"""A transformer that uses the training data provided to fit the model for
	better prediction. This is an abstract trainer, and there are two implementations of it.
	XYTrainer: expects the input to be the data itself in X and Y numpy arrays
	GeneratorTrainer: expects the input to be generators of batches of data

	Input:
		data: training data
		validation_data: validation data

	Output:
		model: trained model

	Args:
		trainer: ``model.Model`` instance to train using data
	"""
    __metaclass__ = ABCMeta
    __out__ = ('model',)

    def __init__(self, name, trainer):
        super(Trainer, self).__init__(name, need_setup=True)
        self.trainer = trainer

    def __setup__(self, **kwargs):
        self.trainer.setup(**kwargs)


class XYTrainer(Trainer):
    def __transform__(self, x, y, validation_data):
        self.log("Training model using x-y data...")
        self.callbacks = self.trainer.create_callbacks(**self.trainer.callbacks_config)
        self.trainer.training_model.fit(x, y,
                                        validation_data=validation_data,
                                        callbacks=self.callbacks,
                                        verbose=1,
                                        **self.trainer.training_config)
        self.log("Trained model")
        return {"model": self.trainer}


class GeneratorTrainer(Trainer):
    def __transform__(self, datagen, validation_datagen):
        self.log("Training model with generator...")
        self.trainer.create_callbacks(**self.trainer.callbacks_config)

        self.trainer.callbacks.append(
            ModelCheckpoint(self.trainer.model, **self.trainer.callbacks_config['model_checkpoint']))
        train_flow, train_steps = datagen
        valid_flow, valid_steps = validation_datagen
        self.trainer.training_model.fit_generator(train_flow,
                                                  steps_per_epoch=train_steps,
                                                  validation_data=valid_flow,
                                                  validation_steps=valid_steps,
                                                  callbacks=self.trainer.callbacks,
                                                  verbose=1,
                                                  **self.trainer.training_config)
        self.log("Trained model")
        return {"model": self.trainer}