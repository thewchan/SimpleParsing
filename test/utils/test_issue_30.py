
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, TypeVar, Union

from simple_parsing import Serializable


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
    with open(path) as f:
        train_options_dict = yaml.safe_load(f)
    network_ = NetworkOptions.from_dict(train_options_dict["network_options"])

    assert network == network_

    # Lastly, here's exactly what you asked for, which you can just copy-paste
    # if you want :)
    network_ = load_flexible(NetworkOptions, path, parse_subpath="network_options")
    assert network == network_

S = TypeVar("S", bound=Serializable)

def load_flexible(cls: Type[S],
                  path: Union[str, Path],
                  parse_subpath: Union[str, List[str]]=None) -> S:
    """Loads an instance of class `cls` from the path at `path`, optionally
    at a subpath inside the json or yaml file.

    Args:
        cls (Type[S]): A Serializable dataclass to instantiate.
        path (Union[str, Path]): Path to load form.
        parse_subpath (Union[str, List[str]], optional): Optional key to load
            from the deserialized dictionary. When `None` (default), loads the
            dataclass from the file. When given a string, loads the dataclass
            from the value at that key in the deserialized dictionary.
            When given a sequence of keys, loads the dataclass from the value
            in the nested dictionary by using the keys in sequence. Defaults to
            None.

    Returns:
        S: The instance of the dataclass type `cls`.
    """
    path = Path(path)
    if parse_subpath is None:
        return cls.load(path)
    
    # We want to parse a sub-entry of the serialized dict.
    import json
    load = json.load
    if path.suffix.endswith((".yml", ".yaml")):
        import yaml
        load = yaml.safe_load

    with open(path) as f:
        parent_dict = load(f)

    if isinstance(parse_subpath, str):
        # we just get the entry directly.
        return cls.from_dict(parent_dict[parse_subpath])

    # The key is a list, meaning its inside a nested dict.
    for k in parse_subpath:
        parent_dict = parent_dict[k]
    # at the end, parent_dict is the dict from which we want to construct cls.
    return cls.from_dict(parent_dict)
