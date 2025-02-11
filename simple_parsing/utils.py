"""Utility functions used in various parts of the simple_parsing package."""
import argparse
import builtins
import collections
import copy
import inspect
import enum
import sys
import dataclasses
import functools
import json
import re
import warnings
import itertools
import hashlib
from abc import ABC
from collections import OrderedDict
from collections import abc as c_abc
from collections import defaultdict
from dataclasses import _MISSING_TYPE, MISSING, Field, dataclass
from enum import Enum
from functools import partial
from inspect import isclass
from logging import getLogger
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import typing
from typing import get_type_hints

# from typing_inspect import get_origin, is_typevar, get_bound, is_forward_ref, get_forward_arg
NEW_TYPING = sys.version_info[:3] >= (3, 7, 0)  # PEP 560

if sys.version_info < (3, 9):
    # TODO: Add 3.9 compatibility, remove typing_inspect dependency.
    from typing_inspect import (
        get_origin,
        is_typevar,
        get_bound,
        get_forward_arg,
        is_forward_ref,
    )
else:
    from typing import get_origin

    # NOTE: Copied over from typing_inspect.
    def is_typevar(t) -> bool:
        return type(t) is TypeVar

    def get_bound(t):
        if is_typevar(t):
            return getattr(t, "__bound__", None)
        else:
            raise TypeError(f"type is not a `TypeVar`: {t}")

    def is_forward_ref(t):
        return isinstance(t, typing.ForwardRef)

    def get_forward_arg(fr):
        return getattr(fr, "__forward_arg__", None)


try:
    from typing import get_args
except ImportError:
    # try:
    #     # TODO: Not sure we should depend on typing_inspect, results appear to vary
    #     # greatly
    #     # between python versions.
    #     from typing_inspect import get_args
    # except ImportError:
    def get_args(some_type: Type) -> Tuple[Type, ...]:
        return getattr(some_type, "__args__", ())


try:
    from typing import get_origin
except ImportError:
    from typing_inspect import get_origin


logger = getLogger(__name__)

builtin_types = [
    getattr(builtins, d)
    for d in dir(builtins)
    if isinstance(getattr(builtins, d), type)
]

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

Dataclass = TypeVar("Dataclass")
DataclassType = Type[Dataclass]

SimpleValueType = Union[bool, int, float, str]
SimpleIterable = Union[
    List[SimpleValueType], Dict[Any, SimpleValueType], Set[SimpleValueType]
]


def is_subparser_field(field: Field) -> bool:
    if is_union(field.type) and not is_choice(field):
        type_arguments = get_type_arguments(field.type)
        return all(map(dataclasses.is_dataclass, type_arguments))
    return bool(field.metadata.get("subparsers", {}))


class InconsistentArgumentError(RuntimeError):
    """
    Error raised when the number of arguments provided is inconsistent when parsing multiple instances from command line.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def camel_case(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


TRUE_STRINGS: List[str] = ["yes", "true", "t", "y", "1"]
FALSE_STRINGS: List[str] = ["no", "false", "f", "n", "0"]


def str2bool(raw_value: Union[str, bool]) -> bool:
    """
    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(raw_value, bool):
        return raw_value
    v = raw_value.strip().lower()
    if v in TRUE_STRINGS:
        return True
    elif v in FALSE_STRINGS:
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected for argument, received '{raw_value}'"
        )


def get_item_type(container_type: Type[Container[T]]) -> T:
    """Returns the `type` of the items in the provided container `type`.

    When no type annotation is found, or no item type is found, returns
    `typing.Any`.
    NOTE: If a type with multiple arguments is passed, only the first type
    argument is returned.

    >>> import typing
    >>> from typing import List, Tuple
    >>> get_item_type(list)
    typing.Any
    >>> get_item_type(List)
    typing.Any
    >>> get_item_type(tuple)
    typing.Any
    >>> get_item_type(Tuple)
    typing.Any
    >>> get_item_type(List[int])
    <class 'int'>
    >>> get_item_type(List[str])
    <class 'str'>
    >>> get_item_type(List[float])
    <class 'float'>
    >>> get_item_type(List[float])
    <class 'float'>
    >>> get_item_type(List[Tuple])
    typing.Tuple
    >>> get_item_type(List[Tuple[int, int]])
    typing.Tuple[int, int]
    >>> get_item_type(Tuple[int, str])
    <class 'int'>
    >>> get_item_type(Tuple[str, int])
    <class 'str'>
    >>> get_item_type(Tuple[str, str, str, str])
    <class 'str'>

    Arguments:
        list_type {Type} -- A type, preferably one from the Typing module (List, Tuple, etc).

    Returns:
        Type -- the type of the container's items, if found, else Any.
    """
    if container_type in {
        list,
        set,
        tuple,
        List,
        Set,
        Tuple,
        Dict,
        Mapping,
        MutableMapping,
    }:
        # the built-in `list` and `tuple` types don't have annotations for their item types.
        return Any
    type_arguments = getattr(container_type, "__args__", None)
    if type_arguments:
        return type_arguments[0]
    else:
        return Any


def get_argparse_type_for_container(
    container_type: Type[Container[T]],
) -> Union[Type[T], Callable[[str], T]]:
    """Gets the argparse 'type' option to be used for a given container type.
    When an annotation is present, the 'type' option of argparse is set to that type.
    if not, then the default value of 'str' is returned.

    Arguments:
        container_type {Type} -- A container type (ideally a typing.Type such as List, Tuple, along with an item annotation: List[str], Tuple[int, int], etc.)

    Returns:
        typing.Type -- the type that should be used in argparse 'type' argument option.

    TODO: This overlaps in a weird way with `get_parsing_fn`, which returns the 'type'
    to use for a given annotation! This function however doesn't deal with 'weird' item
    types, it just returns the first annotation.
    """
    T = get_item_type(container_type)
    if T is bool:
        return str2bool
    if T is Any:
        return str
    if is_enum(T):
        # IDEA: Fix this weirdness by first moving all this weird parsing logic into the
        # field wrapper class, and then split it up into different subclasses of FieldWrapper,
        # each for a different type of field.
        from simple_parsing.wrappers.field_parsing import parse_enum

        return parse_enum(T)
    return T


def _mro(t: Type) -> List[Type]:
    # TODO: This is mostly used in 'is_tuple' and such, and should be replaced with
    # either the built-in 'get_origin' from typing, or from typing-inspect.
    if t is None:
        return []
    if hasattr(t, "__mro__"):
        return t.__mro__
    elif get_origin(t) is type:
        return []
    elif hasattr(t, "mro") and callable(t.mro):
        return t.mro()
    return []


def is_list(t: Type) -> bool:
    """returns True when `t` is a List type.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is list or a subclass of list.

    >>> from typing import *
    >>> is_list(list)
    True
    >>> is_list(tuple)
    False
    >>> is_list(List)
    True
    >>> is_list(List[int])
    True
    >>> is_list(List[Tuple[int, str, None]])
    True
    >>> is_list(Optional[List[int]])
    False
    >>> class foo(List[int]):
    ...   pass
    ...
    >>> is_list(foo)
    True
    """
    return list in _mro(t)


def is_tuple(t: Type) -> bool:
    """returns True when `t` is a tuple type.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is tuple or a subclass of tuple.

    >>> from typing import *
    >>> is_tuple(list)
    False
    >>> is_tuple(tuple)
    True
    >>> is_tuple(Tuple)
    True
    >>> is_tuple(Tuple[int])
    True
    >>> is_tuple(Tuple[int, str, None])
    True
    >>> class foo(tuple):
    ...   pass
    ...
    >>> is_tuple(foo)
    True
    >>> is_tuple(List[int])
    False
    """
    return tuple in _mro(t)


def is_dict(t: Type) -> bool:
    """returns True when `t` is a dict type or annotation.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is dict or a subclass of dict.

    >>> from typing import *
    >>> from collections import OrderedDict
    >>> is_dict(dict)
    True
    >>> is_dict(OrderedDict)
    True
    >>> is_dict(tuple)
    False
    >>> is_dict(Dict)
    True
    >>> is_dict(Dict[int, float])
    True
    >>> is_dict(Dict[Any, Dict])
    True
    >>> is_dict(Optional[Dict])
    False
    >>> is_dict(Mapping[str, int])
    True
    >>> class foo(Dict):
    ...   pass
    ...
    >>> is_dict(foo)
    True
    """
    mro = _mro(t)
    return dict in mro or Mapping in mro or c_abc.Mapping in mro


def is_set(t: Type) -> bool:
    """returns True when `t` is a set type or annotation.

    Args:
        t (Type): a type.

    Returns:
        bool: True if `t` is set or a subclass of set.

    >>> from typing import *
    >>> is_set(set)
    True
    >>> is_set(Set)
    True
    >>> is_set(tuple)
    False
    >>> is_set(Dict)
    False
    >>> is_set(Set[int])
    True
    >>> is_set(Set["something"])
    True
    >>> is_set(Optional[Set])
    False
    >>> class foo(Set):
    ...   pass
    ...
    >>> is_set(foo)
    True
    """
    mro = _mro(t)
    return set in _mro(t)


def is_dataclass_type(t: Type) -> bool:
    """Returns whether t is a dataclass type or a TypeVar of a dataclass type.

    Args:
        t (Type): Some type.

    Returns:
        bool: Whether its a dataclass type.
    """
    return dataclasses.is_dataclass(t) or (
        is_typevar(t) and dataclasses.is_dataclass(get_bound(t))
    )


def is_enum(t: Type) -> bool:
    if inspect.isclass(t):
        return issubclass(t, enum.Enum)
    return Enum in _mro(t)


def is_bool(t: Type) -> bool:
    return bool in _mro(t)


def is_tuple_or_list(t: Type) -> bool:
    return is_list(t) or is_tuple(t)


def is_union(t: Type) -> bool:
    """Returns whether or not the given Type annotation is a variant (or subclass) of typing.Union

    Args:
        t (Type): some type annotation

    Returns:
        bool: Whether this type represents a Union type.

    >>> from typing import *
    >>> is_union(Union[int, str])
    True
    >>> is_union(Union[int, str, float])
    True
    >>> is_union(Tuple[int, str])
    False
    """
    return getattr(t, "__origin__", "") == Union


def is_homogeneous_tuple_type(t: Type[Tuple]) -> bool:
    """Returns whether the given Tuple type is homogeneous: if all items types are the
    same.

    This also includes Tuple[<some_type>, ...]

    Returns
    -------
    bool

    >>> from typing import *
    >>> is_homogeneous_tuple_type(Tuple)
    True
    >>> is_homogeneous_tuple_type(Tuple[int, int])
    True
    >>> is_homogeneous_tuple_type(Tuple[int, str])
    False
    >>> is_homogeneous_tuple_type(Tuple[int, str, float])
    False
    >>> is_homogeneous_tuple_type(Tuple[int, ...])
    True
    >>> is_homogeneous_tuple_type(Tuple[Tuple[int, str], ...])
    True
    >>> is_homogeneous_tuple_type(Tuple[List[int], List[str]])
    False
    """
    if not is_tuple(t):
        return False
    type_arguments = get_type_arguments(t)
    if not type_arguments:
        return True
    assert isinstance(type_arguments, tuple), type_arguments
    if len(type_arguments) == 2 and type_arguments[1] is Ellipsis:
        return True
    # Tuple[str, str, str] -> True
    # Tuple[str, str, float] -> False
    # TODO: Not sure if this will work with more complex item times (like nested tuples)
    return len(set(type_arguments)) == 1


def is_choice(field: Field) -> bool:
    return bool(field.metadata.get("custom_args", {}).get("choices", {}))


def is_optional(t: Type) -> bool:
    """Returns True if the given Type is a variant of the Optional type.

    Parameters
    ----------
    - t : Type

        a Type annotation (or "live" type)

    Returns
    -------
    bool
        Whether or not this is an Optional.

    >>> from typing import Union, Optional, List
    >>> is_optional(str)
    False
    >>> is_optional(Optional[str])
    True
    >>> is_optional(Union[str, None])
    True
    >>> is_optional(Union[str, List])
    False
    >>> is_optional(Union[str, List, int, float, None])
    True
    """
    return is_union(t) and type(None) in get_type_arguments(t)


def is_tuple_or_list_of_dataclasses(t: Type) -> bool:
    return is_tuple_or_list(t) and is_dataclass_type(get_item_type(t))


def contains_dataclass_type_arg(t: Type) -> bool:
    if is_dataclass_type(t):
        return True
    elif is_tuple_or_list_of_dataclasses(t):
        return True
    elif is_union(t):
        return any(contains_dataclass_type_arg(arg) for arg in get_type_arguments(t))
    return False


def get_dataclass_type_arg(t: Type) -> Optional[Type]:
    if not contains_dataclass_type_arg(t):
        return None
    if is_dataclass_type(t):
        return t
    elif is_tuple_or_list(t) or is_union(t):
        return next(
            filter(
                None, (get_dataclass_type_arg(arg) for arg in get_type_arguments(t))
            ),
            None,
        )
    return None


def get_type_arguments(container_type: Type) -> Tuple[Type, ...]:
    # return getattr(container_type, "__args__", ())
    return get_args(container_type)


def get_type_name(some_type: Type):
    result = getattr(some_type, "__name__", str(some_type))
    type_arguments = get_type_arguments(some_type)
    if type_arguments:
        result += f"[{','.join(get_type_name(T) for T in type_arguments)}]"
    return result


def get_container_nargs(container_type: Type) -> Union[int, str]:
    """Gets the value of 'nargs' appropriate for the given container type.

    Parameters
    ----------
    container_type : Type
        Some container type.

    Returns
    -------
    Union[int, str]
        [description]
    """
    if is_tuple(container_type):
        # TODO: Should a `Tuple[int]` annotation be interpreted as "a tuple of an
        # unknown number of ints"?.
        type_arguments: Tuple[Type, ...] = get_type_arguments(container_type)
        if not type_arguments:
            return "*"
        if len(type_arguments) == 2 and type_arguments[1] is Ellipsis:
            return "*"

        total_nargs: int = 0
        for item_type in type_arguments:
            # TODO: Handle the 'nargs' for nested container types!
            if is_list(item_type) or is_tuple(item_type):
                # BUG: If it's a container like Tuple[Tuple[int, str], Tuple[int, str]]
                # we could do one of two things:
                #
                # - Option 1: Use nargs=4 and re-organize/split values in
                #   post-processing.
                # item_nargs: Union[int, str] = get_container_nargs(item_type)
                # if isinstance(item_nargs, int):
                #     total_nargs += item_nargs
                # else:
                #     return "*"
                #
                # This is a bit confusing, and IMO it might be best to just do
                # - Option 2: Use `nargs='*'` and use a custom parsing function that
                #   will convert entries appropriately..
                return "*"
            total_nargs += 1
        return total_nargs

    if is_list(container_type):
        return "*"
    raise NotImplementedError(
        f"Not sure what 'nargs' should be for type {container_type}"
    )


def _parse_multiple_containers(
    container_type: type, append_action: bool = False
) -> Callable[[str], List[Any]]:
    T = get_argparse_type_for_container(container_type)
    factory = tuple if is_tuple(container_type) else list

    result = factory()

    def parse_fn(value: str):
        logger.debug(f"parsing multiple {container_type} of {T}s, value is: '{value}'")
        values = _parse_container(container_type)(value)
        logger.debug(f"parsing result is '{values}'")

        if append_action:
            result += values
            return result
        else:
            return values

    return parse_fn


def _parse_container(container_type: Type[Container]) -> Callable[[str], List[Any]]:
    T = get_argparse_type_for_container(container_type)
    factory = tuple if is_tuple(container_type) else list
    import ast

    result: List[Any] = []

    def _parse(value: str) -> List[Any]:
        logger.debug(f"Parsing a {container_type} of {T}s, value is: '{value}'")
        try:
            values = _parse_literal(value)
        except Exception as e:
            logger.debug(
                f"Exception while trying to parse '{value}' as a literal: {type(e)}: {e}"
            )
            # if it doesn't work, fall back to the parse_fn.
            values = _fallback_parse(value)

        # we do the default 'argparse' action, which is to add the values to a bigger list of values.
        # result.extend(values)
        logger.debug(f"returning values: {values}")
        return values

    def _parse_literal(value: str) -> Union[List[Any], Any]:
        """try to parse the string to a python expression directly.
        (useful for nested lists or tuples.)
        """
        literal = ast.literal_eval(value)
        logger.debug(f"Parsed literal: {literal}")
        if not isinstance(literal, (list, tuple)):
            # we were passed a single-element container, like "--some_list 1", which should give [1].
            # We therefore return the literal itself, and argparse will append it.
            return T(literal)
        else:
            container = literal
            values = factory(T(v) for v in container)
            return values

    def _fallback_parse(v: str) -> List[Any]:
        v = " ".join(v.split())
        if v.startswith("[") and v.endswith("]"):
            v = v[1:-1]

        separator = " "
        for sep in [","]:  # TODO: maybe add support for other separators?
            if sep in v:
                separator = sep

        str_values = [v.strip() for v in v.split(separator)]
        T_values = [T(v_str) for v_str in str_values]
        values = factory(v for v in T_values)
        return values

    _parse.__name__ = T.__name__
    return _parse


def setattr_recursive(obj: object, attribute_name: str, value: Any):
    if "." not in attribute_name:
        setattr(obj, attribute_name, value)
    else:
        parts = attribute_name.split(".")
        child_object = getattr(obj, parts[0])
        setattr_recursive(child_object, ".".join(parts[1:]), value)


def split_dest(destination: str) -> Tuple[str, str]:
    splits = destination.split(".")
    parent = ".".join(splits[:-1])
    attribute_in_parent = splits[-1]
    return parent, attribute_in_parent


def get_nesting_level(possibly_nested_list):
    if not isinstance(possibly_nested_list, (list, tuple)):
        return 0
    elif len(possibly_nested_list) == 0:
        return 1
    else:
        return 1 + max(get_nesting_level(item) for item in possibly_nested_list)


def default_value(field: dataclasses.Field) -> Union[T, _MISSING_TYPE]:
    """Returns the default value of a field in a dataclass, if available.
    When not available, returns `dataclasses.MISSING`.

    Args:
        field (dataclasses.Field): The dataclasses.Field to get the default value of.

    Returns:
        Union[T, _MISSING_TYPE]: The default value for that field, if present, or None otherwise.
    """
    if field.default is not dataclasses.MISSING:
        return field.default
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore
        constructor = field.default_factory  # type: ignore
        return constructor()
    else:
        return dataclasses.MISSING


def trie(sentences: List[List[str]]) -> Dict[str, Union[str, Dict]]:
    """Given a list of sentences, creates a trie as a nested dicts of word strings.

    Args:
        sentences (List[List[str]]): a list of sentences

    Returns:
        Dict[str, Union[str, Dict[str, ...]]]: A tree where each node is a word in a sentence.
        Sentences which begin with the same words share the first nodes, etc.
    """
    first_word_to_sentences: Dict[str, List[List[str]]] = defaultdict(list)
    for sentence in sentences:
        first_word = sentence[0]
        first_word_to_sentences[first_word].append(sentence)

    return_dict: Dict[str, Union[str, Dict]] = {}
    for first_word, sentences in first_word_to_sentences.items():
        if len(sentences) == 1:
            return_dict[first_word] = ".".join(sentences[0])
        else:
            sentences_without_first_word = [sentence[1:] for sentence in sentences]
            return_dict[first_word] = trie(sentences_without_first_word)
    return return_dict


def keep_keys(d: Dict, keys_to_keep: Iterable[str]) -> Tuple[Dict, Dict]:
    """Removes all the keys in `d` that aren't in `keys`.

    Parameters
    ----------
    d : Dict
        Some dictionary.
    keys_to_keep : Iterable[str]
        The set of keys to keep

    Returns
    -------
    Tuple[Dict, Dict]
        The same dictionary (with all the unwanted keys removed) as well as a
        new dict containing only the removed item.

    """
    d_keys = set(d.keys())  # save a copy since we will modify the dict.
    removed = {}
    for key in d_keys:
        if key not in keys_to_keep:
            removed[key] = d.pop(key)
    return d, removed


def compute_identity(size: int = 16, **sample) -> str:
    """Compute a unique hash out of a dictionary

    Parameters
    ----------
    size: int
        size of the unique hash

    **sample:
        Dictionary to compute the hash from

    """
    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode("utf8"))

        if isinstance(v, dict):
            sample_hash.update(compute_identity(size, **v).encode("utf8"))
        else:
            sample_hash.update(str(v).encode("utf8"))

    return sample_hash.hexdigest()[:size]


def dict_intersection(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[V, ...]]]:
    common_keys = set(dicts[0])
    for d in dicts:
        common_keys.intersection_update(d)
    for key in common_keys:
        yield (key, tuple(d[key] for d in dicts))


def field_dict(dataclass: Dataclass) -> Dict[str, Field]:
    result: Dict[str, Field] = OrderedDict()
    for field in dataclasses.fields(dataclass):
        result[field.name] = field
    return result


def zip_dicts(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[Optional[V], ...]]]:
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    keys = set(itertools.chain(*dicts))
    for key in keys:
        yield (key, tuple(d.get(key) for d in dicts))


def dict_union(
    *dicts: Dict[K, V], recurse: bool = True, dict_factory=dict
) -> Dict[K, V]:
    """Simple dict union until we use python 3.9

    If `recurse` is True, also does the union of nested dictionaries.
    NOTE: The returned dictionary has keys sorted alphabetically.
    >>> from collections import OrderedDict
    >>> a = OrderedDict(a=1, b=2, c=3)
    >>> b = OrderedDict(c=5, d=6, e=7)
    >>> dict_union(a, b, dict_factory=OrderedDict)
    OrderedDict([('a', 1), ('b', 2), ('c', 5), ('d', 6), ('e', 7)])
    >>> a = OrderedDict(a=1, b=OrderedDict(c=2, d=3))
    >>> b = OrderedDict(a=2, b=OrderedDict(c=3, e=6))
    >>> dict_union(a, b, dict_factory=OrderedDict)
    OrderedDict([('a', 2), ('b', OrderedDict([('c', 3), ('d', 3), ('e', 6)]))])
    """
    result: Dict = dict_factory()
    if not dicts:
        return result
    assert len(dicts) >= 1
    all_keys: Set[str] = set()
    all_keys.update(*dicts)
    all_keys = sorted(all_keys)

    # Create a neat generator of generators, to save some memory.
    all_values: Iterable[Tuple[V, Iterable[K]]] = (
        (k, (d[k] for d in dicts if k in d)) for k in all_keys
    )
    for k, values in all_values:
        sub_dicts: List[Dict] = []
        new_value: V = None
        n_values = 0
        for v in values:
            if isinstance(v, dict) and recurse:
                sub_dicts.append(v)
            else:
                # Overwrite the new value for that key.
                new_value = v
            n_values += 1

        if len(sub_dicts) == n_values and recurse:
            # We only get here if all values for key `k` were dictionaries,
            # and if recurse was True.
            new_value = dict_union(*sub_dicts, recurse=True, dict_factory=dict_factory)

        result[k] = new_value
    return result


# NOTE: This dict is used to enable forward compatibility with things such as `tuple[int, str]`,
# `list[float]`, etc. when using `from __future__ import annotations`.
forward_refs_to_types = {
    "tuple": typing.Tuple,
    "set": typing.Set,
    "dict": typing.Dict,
    "list": typing.List,
    "type": Type,
}


def get_field_type_from_annotations(some_class: type, field_name: str) -> type:
    """
    If the script uses `from __future__ import annotations`, and we are in python<3.9,
    Then we need to actually first make this forward-compatibility 'patch' so that we
    don't run into a "`type` object is not subscriptable" error.

    NOTE: If you get errors of this kind from the function below, then you might want to add an
    entry to the `forward_refs_to_types` dict above.
    """
    # The type of the field might be a string when using `from __future__ import annotations`.
    # Get the local and global namespaces to pass to the `get_type_hints` function.
    local_ns: Dict[str, Any] = {"typing": typing, **vars(typing)}
    if sys.version_info < (3, 9):
        local_ns.update(forward_refs_to_types)
    global_ns = sys.modules[some_class.__module__].__dict__
    try:
        class_type_hints = get_type_hints(
            some_class, localns=local_ns, globalns=global_ns
        )
        field_type = class_type_hints[field_name]

        if sys.version_info >= (3, 10):
            # In python >= 3.10, int | float is allowed. Therefore, just to be consistent, we want
            # to convert those into the corresponding typing.Union type.
            args = typing.get_args(field_type)
            new_args = []

            import types

            def _replace_UnionTypes_with_typing_Union(annotation):
                if isinstance(annotation, types.UnionType):
                    union_args = typing.get_args(annotation)
                    new_union_args = tuple(
                        _replace_UnionTypes_with_typing_Union(arg) for arg in union_args
                    )
                    return typing.Union[new_union_args]
                if is_list(annotation):
                    item_annotation = typing.get_args(annotation)[0]
                    new_item_annotation = _replace_UnionTypes_with_typing_Union(
                        item_annotation
                    )
                    return typing.List[new_item_annotation]
                if is_tuple(annotation):
                    item_annotations = typing.get_args(annotation)
                    new_item_annotations = tuple(
                        _replace_UnionTypes_with_typing_Union(arg)
                        for arg in item_annotations
                    )
                    return typing.Tuple[new_item_annotations]
                if is_dict(annotation):
                    annotations = typing.get_args(annotation)
                    if not annotations:
                        return typing.Dict
                    assert len(annotations) == 2
                    key_annotation = annotations[0]
                    value_annotation = annotations[1]
                    new_key_annotation = _replace_UnionTypes_with_typing_Union(
                        key_annotation
                    )
                    new_value_annotation = _replace_UnionTypes_with_typing_Union(
                        value_annotation
                    )
                    return typing.Dict[new_key_annotation, new_value_annotation]
                if annotation in builtin_types:
                    return annotation
                if inspect.isclass(annotation):
                    return annotation
                raise NotImplementedError(annotation)

            new_field_type = _replace_UnionTypes_with_typing_Union(field_type)
            field_type = new_field_type

    except TypeError as err:
        annotations_dict = some_class.__annotations__.copy()
        if any(
            isinstance(annotation, str) and "|" in annotation
            for annotation in annotations_dict.values()
        ):
            # Pretty hacky: Modify the type annotations of the class (preferably a copy of the class
            # if possible, to avoid modifying things in-place), and replace  the `a | b`-type
            # expressions with `Union[a, b]`, so that `get_type_hints` doesn't raise an error.
            try:
                before = some_class.__annotations__.copy()
                after = _replace_new_union_syntax_with_old_union_syntax(
                    before, local_ns=local_ns, global_ns=global_ns
                )

                class _Temp:
                    pass

                _Temp.__annotations__ = after
                class_type_hints = get_type_hints(
                    _Temp, localns=local_ns, globalns=global_ns
                )
                field_type = class_type_hints[field_name]
                return field_type
            except (TypeError, NotImplementedError) as exc:
                print(err)
                err = exc
                pass
        annotation = annotations_dict[field_name]
        logger.error(
            f"Error when parsing type annotations of class {some_class}: {err}\n"
            f"Using a 'str' type for the field instead of the original type: {annotation}).\n"
            f"Please make an issue at https://www.github.com/lebrice/SimpleParsing/issues "
            f"if you believe this annotation should be supported explicitly."
        )
        return str
    if sys.version_info[:2] >= (3, 7):
        # Weird bug happens when mixing postponed evaluation of type annotations + forward
        # references: The ForwardRefs are left as-is, and not evaluated!
        from typing import ForwardRef

        if isinstance(field_type, ForwardRef):
            forward_arg = field_type.__forward_arg__
            if forward_arg in global_ns:
                field_type = global_ns[forward_arg]
            else:
                logger.warning(
                    f"Unable to evaluate forward reference {field_type} for field '{field_name}'.\n"
                    f"Leaving it as-is."
                )
    return field_type


def _replace_new_union_syntax_with_old_union_syntax(
    annotations_dict: Dict[str, str], local_ns: dict, global_ns: dict
) -> Dict[str, Any]:
    # Pretty hacky: Modify the type annotations of the class (preferably a copy of the class
    # if possible, to avoid modifying things in-place), and replace  the `a | b`-type
    # expressions with `Union[a, b]`, so that `get_type_hints` doesn't raise an error.
    # The type of the field might be a string when using `from __future__ import annotations`.

    import builtins

    def _get_type(ann: str) -> Union[type, str]:
        # Try locals, then globals, then builtins. Otherwise, use the annotation itself.
        maps = collections.ChainMap(
            forward_refs_to_types, local_ns, global_ns, builtins.__dict__
        )
        return maps.get(ann, ann)
        # return forward_refs_to_types.get(ann, local_ns.get(ann, global_ns.get(ann, getattr(builtins, ann, ann))))

    def _not_supported() -> typing.NoReturn:
        raise NotImplementedError(
            f"Don't yet support annotations like these: {annotations_dict}"
        )

    def _get_old_style_annotation(annotation: str) -> str:
        # TODO: Add proper support for things like `list[int | float]`, which isn't currently
        # working, even without the new-style union.
        if "|" not in annotation:
            return annotation

        annotation = annotation.strip()
        if "[" not in annotation:
            assert "]" not in annotation
            return "Union[" + ", ".join(v.strip() for v in annotation.split("|")) + "]"

        before, lsep, rest = annotation.partition("[")
        middle, rsep, after = rest.rpartition("]")
        assert (
            not after.strip()
        ), "can't have text at HERE in <something>[<something>]<HERE>!"

        if "|" in before or "|" in after:
            _not_supported()
        assert "|" in middle

        if "," in middle:
            parts = [v.strip() for v in middle.split(",")]
            parts = [_get_old_style_annotation(part) for part in parts]
            middle = ", ".join(parts)

        new_middle = _get_old_style_annotation(annotation=middle)
        new_annotation = str(_get_type(before)) + lsep + new_middle + rsep + after
        return new_annotation

    new_annotations = annotations_dict.copy()
    for field, annotation_str in annotations_dict.items():
        updated_annotation = _get_old_style_annotation(annotation_str)
        new_annotations[field] = updated_annotation

    return new_annotations


if __name__ == "__main__":
    import doctest

    doctest.testmod()
