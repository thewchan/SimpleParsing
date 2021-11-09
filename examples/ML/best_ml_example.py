from typing import List, Optional
from typing import TypeVar
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from dataclasses import asdict
""" Best practices example for an ML project with SimpleParsing.
"""
import random
import dataclasses
import logging
from typing import Type, Union
from simple_parsing import ArgumentParser
import json
import wandb
from dataclasses import dataclass
from abc import ABC, abstractmethod
from simple_parsing.helpers.fields import choice, list_field
from simple_parsing.helpers.hparams import HyperParameters, log_uniform, uniform
from torch import nn

from simple_parsing.helpers.hparams.hparam import categorical
from typing import Callable, Tuple
import wandb
from torch import Tensor
import numpy as np
from pathlib import Path
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Classifier(ABC, nn.Module):
    """ Example of an ABC for a classifier model. Other models are created as variants of this one. """
    @dataclass
    class HParams(HyperParameters):
        """ Hyper-Parameters of a classifier. """
        # Learning rate.
        learning_rate: float = log_uniform(1e-7, 1e-2, default=3e-4)

        # batch size
        batch_size: int = log_uniform(16, 512, default=64, discrete=True)

        # Type of activation to use.
        activation: Type[nn.Module] = categorical({
            "elu": nn.ELU,
            "relu6": nn.ReLU6,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }, default=nn.ReLU)

    def __init__(self, input_dims: Tuple[int, ...], output_dims: int, hparams: "Classifier.HParams" = None):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        # Sample from the prior when doing HPO (when wandb.run is set), otherwise use
        # the default values.
        self.hparams = hparams or (
            self.HParams.sample() if wandb.run else self.HParams())

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """ Return the classification logits (log-probabilities) for a given batch. """


class RandomBaseline(Classifier):
    """ Random baseline classifier. """
    def forward(self, input: Tensor) -> Tensor:
        return torch.randn([input.shape[0]], dtype=input.dtype, device=input.device)


class MLP(Classifier):
    """ A Multi-Layer Perceptron classifier. """

    @dataclass
    class HParams(Classifier.HParams):
        """ Hyper-parameters for an MLP classifier. """
        # Number of neurons to use per layer.
        neurons_per_layer: List[int] = list_field(32, 64, 64)

    def __init__(self, input_dims: Tuple[int, ...], output_dims: int, hparams: "MLP.HParams" = None):
        super().__init__(input_dims=input_dims, output_dims=output_dims, hparams=hparams)
        self.hparams: MLP.HParams

        # Create the layers of the network.
        layers: List[nn.Module] = []
        layers.append(nn.Flatten())

        in_features = np.product(self.input_dims)
        neurons: List[int] = [in_features, *self.hparams.neurons_per_layer]
        # middle layers
        for i, (in_features, out_features) in enumerate(zip(neurons[0:-1], neurons[1:])):
            layers.append(nn.Linear(in_features=in_features,
                                    out_features=out_features))
            layers.append(self.hparams.activation())
        # last layer
        layers.append(
            nn.Linear(in_features=neurons[-1], out_features=output_dims))
        self.net = nn.Sequential(layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class CNN(Classifier):
    """ Classifier that uses a Convolutional Neural Network (CNN). """

    @dataclass
    class HParams(Classifier.HParams):
        """ Hyper-parameters for a CNN classifier. """
        # Number of channels per convolutional layer.
        channels_per_layer: List[int] = list_field(32, 64, 64)

        kernel_size: int = 3
        padding: int = 1

    def __init__(self, input_dims: Tuple[int, ...], output_dims: int, hparams: "CNN.HParams" = None):
        super().__init__(input_dims=input_dims, output_dims=output_dims, hparams=hparams)
        self.hparams: CNN.HParams

        # Create the layers of the network.
        layers: List[nn.Module] = []
        layers.append(nn.Flatten())

        # assuming that the input is channels-first for now.
        in_channels = input_dims[0]
        channels: List[int] = [in_channels, *self.hparams.channels_per_layer]
        # middle layers
        for i, (in_channels, out_channels) in enumerate(zip(channels[0:-1], channels[1:])):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=self.hparams.kernel_size, padding=self.hparams.padding))
            layers.append(self.hparams.activation())
        # last layer: Use LazyLinear so we don't have to calculate the number of features
        # in advance.
        layers.append(nn.LazyLinear(out_features=output_dims))
        self.net = nn.Sequential(layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, FakeData, CIFAR10, CIFAR100, VisionDataset



@dataclass
class TrainingConfig:
    # Which dataset to use.
    dataset: Type[VisionDataset] = choice({
        "mnist": MNIST,
        "fashion_mnist": FashionMNIST,
        "fake_data": FakeData,
        "cifar_10": CIFAR10,
        "cifar_100": CIFAR100,
    }, default=MNIST)

    # Data where the logs will be stored.
    log_dir: Path = Path("logs")

    # Data where the datasets should be downloaded.
    data_dir: Path = Path("data")
    
    # Random seed.
    seed: Optional[int] = 123

    # Maximum number of epochs.
    max_epochs: int = 10

    # Wether to run in debug mode. Enables more verbose logging and prevents logging stuff
    # to wandb.
    debug: bool = False

    def make_dataset(self) -> VisionDataset:
        """ Create the train, validation, and test datasets. """
        if self.dataset is FakeData:
            dataset = FakeData()
        else:
            # Not considering transforms for now.
            dataset = self.dataset(root=self.data_dir, download=True)
        return dataset


def get_name(model_type: Type[Classifier]) -> str:
    name = model_type.__qualname__
    if name.endswith("Classifier"):
        # Remove the "Classifier suffix"
        name = name.rpartition("Classifier")[0]
    return name


def main():
    """ Main script. """
    parser = ArgumentParser(description=__doc__)
    # Set the random baseline as the default model to use.
    # parser.set_defaults(model_type=MLPClassifier)

    # NOTE: if args for config are added here, then command becomes
    # python main.py (config args) [mlp|cnn,...] (model ags)
    # parser.add_arguments(Config, dest="config")

    subparsers = parser.add_subparsers(
        title="model", description="Type of model to use.", required=True, metavar="model_type"
    )

    for model_type in Classifier.__subclasses__():
        name: str = model_type.__qualname__
        model_parser: ArgumentParser = subparsers.add_parser(
            name, help=f"Use the {name} model.", description=model_type.__doc__
        )
        model_parser.add_arguments(model_type.HParams, "hparams")
        model_parser.set_defaults(model_type=model_type)
        # model_parser.add_arguments(TrainingConfig, dest="config")
    
    # NOTE: Small bug in argparse, we have to set the metavar manually for now.
    subparsers.metavar = "{" + ",".join(t.__qualname__ for t in Classifier.__subclasses__()) + "}"

    # Add the arguments for the training config. (NOTE: Coudld also add to each subparser
    # above) 
    parser.add_arguments(TrainingConfig, dest="config")

    args = parser.parse_args()

    hparams = args.hparams
    training_config: TrainingConfig = args.config
    model_type: Type[Classifier] = args.model_type

    print("HParams:")
    print(hparams.dumps_yaml(indent=1), "\t")
    print("Config:")
    print(training_config.dumps_yaml(indent=1), "\t")
    
    # Seeding
    print(f"Selected seed: {training_config.seed}")
    torch.random.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    random.seed(training_config.seed)

    dataset: VisionDataset = training_config.make_dataset()
    input_dims = np.array(dataset[0][0]).shape
    output_dims = dataset.num_classes  # TODO: Doesn't work with some annoying datasets.

    if training_config.debug:
        print(f"Setting the max_epochs to 1, since '--debug' was passed.")
        hparams.max_epochs = 1
        logger.setLevel(logging.DEBUG)

    # Create the model
    model: Union[BaselineModel, DTP, ParallelDTP] = model_class(
        datamodule=datamodule, hparams=hparams, config=config
    )

    # --- Create the trainer.. ---
    # NOTE: Now each algo can customize how the Trainer gets created.
    trainer = model.create_trainer()

    # --- Run the experiment. ---
    trainer.fit(model, datamodule=datamodule)

    # Run on the test set:
    test_results = trainer.test(model, datamodule=datamodule, verbose=True)

    wandb.finish()
    print(test_results)
    test_accuracy: float = test_results[0]["test/accuracy"]
    return test_accuracy


if __name__ == "__main__":
    main()
