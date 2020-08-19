
from simple_parsing import Serializable
from typing import Tuple
from dataclasses import dataclass

@dataclass
class NetworkOptions(Serializable):
    num_layers: int
    output_dimension: int
    input_dimensions: Tuple[int, int]


def test_issue_30(tmpdir):
    network = NetworkOptions(num_layers=10, output_dimension=4, input_dimensions=(16, 16))
    print(network.to_dict())
    network.save(f"{tmpdir}/train-settings.yaml")
    network_ = NetworkOptions.load_yaml(f"{tmpdir}/train-settings.yaml")

    assert network == network_