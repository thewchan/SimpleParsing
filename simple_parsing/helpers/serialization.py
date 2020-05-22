
import dataclasses
import json
import logging
from collections import OrderedDict, defaultdict
from functools import singledispatch
from dataclasses import dataclass, is_dataclass, fields, Field
from pathlib import Path
from typing import *
from typing import IO
import warnings
import typing_inspect
from typing_inspect import is_generic_type, is_optional_type, get_args

from ..utils import Dataclass
import os
logger = logging.getLogger(str(Path(__file__).absolute()))

D = TypeVar("D", bound="JsonSerializable")


class SimpleEncoder(json.JSONEncoder):
    def default(self, o: Any):
        return encode(o)


@singledispatch
def encode(obj: Any) -> Union[Dict, List, int, str, bool, None]:
    """ Encodes an object into a json-compatible primitive type.
    
    This is used as the 'default' keyword argument to `json.dumps` and
    `json.dump`, and is called when an object is encountered that `json` doesn't
    know how to serialize. 
    
    To register a type as JsonSerializable, you can just register a custom
    serialization function. (There should be no need to do it for dataclasses, 
    since that is supported by this function), use @encode.register (see the docs for singledispatch).    
    """
    try:
        if is_dataclass(obj):
            d: Dict = dict()
            for field in fields(obj):
                value = getattr(obj, field.name)
                d[field.name] = encode(value) 
            return d
        # elif isinstance(obj, Mapping):
        #     return {
        #         k: encode(val) for k, val in obj.items() 
        #     }
        # elif hasattr(obj, "__dict__"):
        #     return obj.__dict__
        else:
            return obj
    except Exception as e:
        logger.debug(f"Cannot encode object {obj}: {e}")
        raise e

@dataclass
class JsonSerializable:
    """
    Enables reading and writing a Dataclass to a JSON file.

    >>> from dataclasses import dataclass
    >>> from simple_parsing.helpers import JsonSerializable
    >>> @dataclass
    ... class Config(JsonSerializable):
    ...   a: int = 123
    ...   b: str = "456"
    ... 
    >>> config = Config()
    >>> config
    Config(a=123, b='456')
    >>> config.save_json("config.json")
    >>> config_ = Config.load_json("config.json")
    >>> config_
    Config(a=123, b='456')
    >>> assert config == config_
    >>> import os
    >>> os.remove("config.json")
    """
    subclasses: ClassVar[List[Type["JsonSerializable"]]] = []
    
    # decode_into_subclasses: ClassVar[Dict[Type["JsonSerializable"], bool]] = defaultdict(bool)
    decode_into_subclasses: ClassVar[bool] = False

    def __init_subclass__(cls, decode_into_subclasses: bool=None):
        logger.info(f"Registering a new JsonSerializable subclass: {cls}")
        super().__init_subclass__()
        if decode_into_subclasses is None:
            # if decode_into_subclasses is None, we will use the value of the
            # parent class, if it is also a subclass of JsonSerializable.
            # Skip the class itself as well as object.
            parents = cls.mro()[1:-1] 
            logger.debug(f"parents: {parents}")

            for parent in parents:
                if parent in JsonSerializable.subclasses:
                    assert issubclass(parent, JsonSerializable)
                    decode_into_subclasses = parent.decode_into_subclasses
                    logger.debug(f"Parent class {parent} has decode_into_subclasses = {decode_into_subclasses}")
                    break
        
        cls.decode_into_subclasses = decode_into_subclasses or False
        if cls not in JsonSerializable.subclasses:
            JsonSerializable.subclasses.append(cls)

    def to_dict(self) -> Dict:
        """ Serializes this dataclass to a dict. """
        # NOTE: This is better than using `dataclasses.asdict` when there are 'Tensor' fields, since those don't
        # support the deepcopy protocol.
        return json.loads(json.dumps(self, default=encode))
    
    @classmethod
    def from_dict(cls: Type[D], obj: Dict, drop_extra_fields: bool=None) -> D:
        """ Parses an instance of `cls` from the given dict.
        
        NOTE: If the `decode_into_subclasses` class attribute is set to True (or
        if `decode_into_subclasses=True` was passed in the class definition),
        then if there are keys in the dict that aren't fields of the dataclass,
        this will decode the dict into an instance the first subclass of `cls`
        which has all required field names present in the dictionary.
        
        Passing `drop_extra_fields=None` (default) will use the class attribute
        described above.
        Passing `drop_extra_fields=True` will decode the dict into an instance
        of `cls` and drop the extra keys in the dict.
        Passing `drop_extra_fields=False` forces the above-mentioned behaviour.
        """
        drop_extra_fields = drop_extra_fields or not cls.decode_into_subclasses
        return from_dict(cls, obj, drop_extra_fields=drop_extra_fields)
    
    def dump(self, fp: IO[str], **dump_kwargs) -> None:
        dump_kwargs.setdefault("cls", SimpleEncoder)
        json.dump(self.to_dict(), fp, **dump_kwargs)

    def dumps(self, **dumps_kwargs) -> str:
        dumps_kwargs.setdefault("cls", SimpleEncoder)
        return json.dumps(self, **dumps_kwargs)
    
    @classmethod
    def load(cls: Type[D], fp: IO[str], drop_extra_fields: bool=None, **load_kwargs) -> D:
        return cls.from_dict(json.load(fp, **load_kwargs), drop_extra_fields=drop_extra_fields)
    
    @classmethod
    def loads(cls: Type[D], s: str, drop_extra_fields: bool=None, **loads_kwargs) -> D:
        return cls.from_dict(json.loads(s, **loads_kwargs), drop_extra_fields=drop_extra_fields)

    def save_json(self, path: Union[str, Path], **dump_kwargs) -> None:
        with open(path, "w") as fp:
            self.dump(fp, **dump_kwargs)
    
    @classmethod
    def load_json(cls: Type[D], path: Union[str, Path], **load_kwargs) -> D:
        with open(path) as fp:
            return cls.load(fp, **load_kwargs)


T = TypeVar("T")
decoding_fns: Dict[Type, Callable[[Any], Any]] = {}


def is_list_type(t: Type) -> bool:
    return issubclass(t, (list, List))


def is_dict_type(t: Type) -> bool:
    return issubclass(t, (dict, Dict, Mapping))


def contains_dataclass_type(t: Type) -> Tuple[bool, Type]:
    if typing_inspect.is_forward_ref(t):
        t = get_dataclass_type_from_forward_ref(t)         
    
    arg = get_first_non_None_type(t)
    if is_dataclass(t):
        return t
    if arg is not None:
        return True, arg
    if typing_inspect.is_union_type(t):
        logger.debug(f"{t} is a Union")
        args = typing_inspect.get_args(t)
        print(args)


def get_dataclass_type_from_forward_ref(forward_ref: Type) -> Type:
    arg = typing_inspect.get_forward_arg(forward_ref)
    potential_classes: List[Type[JsonSerializable]] = [] 
    for serializable_class in JsonSerializable.subclasses:
        if serializable_class.__name__ == arg:
            potential_classes.append(serializable_class)
    return serializable_class
    if not potential_classes:
        warnings.warn(UserWarning(
            f"Unable to find a corresponding type for forward ref "
            f"{forward_ref} inside the registered JsonSerializable subclasses. "
            f"(Consider adding JsonSerializable as a base class to <{arg}>? )."
        ))
        return None
    elif len(potential_classes) > 1:
        warnings.warn(UserWarning(
            f"More than one potential JsonSerializable subclass was found for "
            f"forward ref '{forward_ref}'. The appropriate dataclass will be "
            f"selected based on the matching fields. "
        ))
        return JsonSerializable
    else:
        assert len(potential_classes) == 1
        return potential_classes[0]


def from_dict(cls: Type[Dataclass], d: Dict[str, Any], drop_extra_fields: bool=None) -> Dataclass:
    if d is None:
        return None
    obj_dict: Dict[str, Any] = d.copy()

    init_args: Dict[str, Any] = {}
    non_init_args: Dict[str, Any] = {}
    
    if drop_extra_fields is None:
        logger.debug(f"drop_extra_fields is None, using the on the class {cls}")
        drop_extra_fields = not getattr(cls, "decode_into_subclasses", False)
    
    logger.debug(f"from_dict called with cls {cls}, drop extra fields: {drop_extra_fields}")

    for field in fields(cls):
        name = field.name
        field_type = field.type
        t = contains_dataclass_type(field_type)

        if typing_inspect.is_union_type(field_type):
            logger.debug(f"field has union type: {field_type}")
            a = get_first_non_None_type(field_type)
            logger.debug(f"First non-none type: {a}")
            if a is not None:
                field_type = a

        if typing_inspect.is_forward_ref(field_type):
            logger.debug(f"field_type {field_type} is a forward ref.")
            field_type = get_dataclass_type_from_forward_ref(field_type)
            logger.debug(f"Found the corresponding type: {field_type}")

        if name not in obj_dict:
            logger.warning(f"Couldn't find the field '{name}' in the dict {d}")
            continue
        
        # if name == "losses":
            # print(field.type)
            # args = typing_inspect.get_args(field.type, evaluate=True)
            # value_type = args[1]
            # if typing_inspect.is_forward_ref(value_type):
            #     print(typing_inspect.get_forward_arg(value_type))
            #     print(value_type.__forward_value__)
            #     print(value_type.__forward_code__)
            # print(vars(args))
            # exit()

        field_value = obj_dict.pop(name)

        if field_type in decoding_fns:
            decoding_function = decoding_fns[field_type]
            field_value = decoding_function(field_value)

        elif is_dataclass(field_type):
            field_value = from_dict(field_type, field_value)
        
        elif is_list_type(field_type):
            item_type = get_list_item_type(field_type)
            logger.debug(f"{field_type} is a List[{item_type}]")
            if item_type and contains_dataclass_type(item_type):    
                new_field_list_value: List = []
                for item_args in field_value:
                    # Parse an instance from the dict.
                    if isinstance(item_args, dict):
                        item_instance = from_dict(item_type, item_args)
                        new_field_list_value.append(item_instance)
                    else:
                        new_field_list_value.append(item_args)
                field_value = new_field_list_value
        
        elif is_dict_type(field_type):
            key_type, value_type = get_key_and_value_types(field_type)
            logger.debug(f"{field_type} is a Dict[{key_type}, {value_type}]")
            if contains_dataclass_type(value_type):
                new_field_dict_value: Dict = {}
                for k, item_args in field_value.items():
                    # Parse an instance from the dict.
                    if isinstance(item_args, dict):
                        item_instance = from_dict(value_type, item_args)
                        new_field_dict_value[k] = item_instance
                    else:
                        new_field_dict_value[k] = item_args
                field_value = new_field_dict_value

        if field.init:
            init_args[name] = field_value
        else:
            non_init_args[name] = field_value
    
    extra_args = obj_dict
    
    if extra_args:
        if drop_extra_fields:
            logger.warning(f"Dropping extra args {extra_args}")
            extra_args.clear()
        
        elif issubclass(cls, JsonSerializable):
            # Use the first JsonSerializable derived class that has all the required fields.
            logger.debug(f"Missing field names: {extra_args.keys()}")  

            # Find all the "registered" subclasses of `cls`. (from JsonSerializable)
            derived_classes: List[Type[JsonSerializable]] = []
            for subclass in JsonSerializable.subclasses:
                if issubclass(subclass, cls) and subclass is not cls:
                    derived_classes.append(subclass)
            logger.debug(f"All JsonSerializable derived classes of {cls} available: {derived_classes}")

            from itertools import chain
            # All the arguments that the dataclass should be able to accept in its 'init'.
            req_init_field_names = set(chain(extra_args, init_args))
            
            # Sort the derived classes by their number of init fields, so we choose the first one that fits.
            derived_classes.sort(key=lambda dc: len(get_init_fields(dc)))

            for child_class in derived_classes:
                logger.debug(f"class name: {child_class.__name__}, mro: {child_class.mro()}") 
                child_init_fields: Dict[str, Field] = get_init_fields(child_class)
                child_init_field_names = set(child_init_fields.keys())
                
                if child_init_field_names >= req_init_field_names:
                    logger.debug(f"Using child class {child_class} instead of {cls}, since it has all the required fields.")
                    cls = child_class
                    break
        else:
            raise RuntimeError(f"Unexpected arguments for class {cls}: {extra_args}")

    instance = cls(**init_args, **extra_args)  # type: ignore
    for name, value in non_init_args.items():
        logger.debug(f"Setting non-init field '{name}' on the instance.")
        setattr(instance, name, value)
    return instance


def get_key_and_value_types(dict_type: Type[Dict]) -> Tuple[Optional[Type], Optional[Type]]:
    args = get_args(dict_type)
    if len(args) != 2:
        logger.debug(f"Weird.. the type {dict_type} doesn't have 2 args: {args}")
        return None, None
    K_ = args[0]
    V_ = args[1]

    if typing_inspect.is_forward_ref(V_):
        logger.debug(f"Value is a forward ref type: {V_}")
        V_ = get_dataclass_type_from_forward_ref(V_)

    logger.debug(f"K_: {K_}, V_: {V_}")
    if isinstance(V_, tuple):
        V_ = get_first_non_None_type(V_)
    elif typing_inspect.is_optional_type(V_):
        logger.debug(f"V_ is optional: {V_}")
        V_ = get_first_non_None_type(V_)
    return K_, V_


def get_list_item_type(list_type: Type[List]) -> Optional[Type]:
    logger.debug(f"list type: {list_type}")
    args = get_args(list_type)
    logger.debug(f"args = {args}")
    
    if not args:
        return None

    assert isinstance(args, tuple), args
    if isinstance(args[0], tuple):
        args = args[0]
    assert isinstance(args, tuple), args
    logger.debug(f"args tuple: {args}")
    V_ = get_first_non_None_type(args)
    logger.debug(f"item type: {V_}")

    if typing_inspect.is_optional_type(V_):
        logger.debug(f"item type is still optional!")
        V_ = get_first_non_None_type(V_)
        logger.debug(f"Args (once again..) {V_}")

    if typing_inspect.is_forward_ref(V_):
        V_ = get_dataclass_type_from_forward_ref(V_)

    assert not isinstance(V_, tuple), V_
    return V_


def get_init_fields(dataclass: Type) -> Dict[str, Field]:
    result: Dict[str, Field] = {}
    for field in fields(dataclass):
        if field.init:
            result[field.name] = field
    return result


def get_first_non_None_type(optional_type: Union[Type, Tuple[Type, ...]]) -> Optional[Type]:
    if not isinstance(optional_type, tuple):
        optional_type = get_args(optional_type)
    for arg in optional_type:
        if arg is not Union and arg is not type(None):
            logger.debug(f"arg: {arg} is not union? {arg is not Union}")
            logger.debug(f"arg is not type(None)? {arg is not type(None)}")
            return arg
    return None


def is_dataclass_or_optional_dataclass_type(t: Type) -> bool:
    """ Returns wether `t` is a dataclass type or an Optional[<dataclass type>].
    """
    return is_dataclass(t) or (is_optional_type(t) and
        is_dataclass(get_args(t)[0])
    )
