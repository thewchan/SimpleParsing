
from simple_parsing import Serializable
from typing import Tuple
from dataclasses import dataclass

@dataclass
class NetworkOptions(Serializable):
    num_layers: int
    output_dimension: int
    input_dimensions: Tuple[int, int]


@dataclass
class TrainingScriptOptions(Serializable):
    learning_rate: float
    network_options: NetworkOptions


def test_issue_30(tmpdir):
    network = NetworkOptions(
        num_layers=10,
        output_dimension=4,
        input_dimensions=(16, 16),
    )
    training_options = TrainingScriptOptions(
        learning_rate=3e-4,
        network_options=network,
    )

    path: str = f"{tmpdir}/train-settings.yaml"

    # Save the 'bigger' object somewhere:
    training_options.save(path)
    
    # You're right, you either have to:
    # 1. Load the parent object and then get the right attribute:
    training_options_ = TrainingScriptOptions.load_yaml(path)
    network_ = training_options_.network_options
    assert network == network_

    # OR:
    # 2. Load the yaml file manually and parse the child from the dict entry:
    import yaml
    with open(f"{tmpdir}/train-settings.yaml") as f:
        train_options_dict = yaml.safe_load(f)
    network_ = NetworkOptions.from_dict(train_options_dict["network_options"])

    assert network == network_