from abc import ABCMeta
from model.connections import Transformer
from model.utils import ModelCheckpoint


class Trainer(Transformer):
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
    __out__ = ('model',)

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
