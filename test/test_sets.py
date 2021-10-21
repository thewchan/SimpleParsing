""" Test set fields. """

from .testutils import TestSetup

from dataclasses import dataclass
from typing import Set, Type, List
import pytest
import sys
from simple_parsing.helpers import field


@pytest.mark.parametrize("item_type, values",[
    (str, ["1", "bob", "clarice"]),
    (int, [1, 2, 3]),
    (float, [1., 2., 3.]),
])
def test_set_field(item_type: Type, values: List):
    @dataclass
    class Foo(TestSetup):
        a: Set[item_type] = field(type=item_type, action="append", nargs="*")  # type: ignore

    assert Foo.setup("--a " + " ".join(values)) == Foo(a=set(values))


@pytest.mark.skipif(sys.version_info < (3, 9))
def test_lowercase_set_field():
    @dataclass
    class Foo(TestSetup):
        a: set[str]

    assert Foo.setup("--a 1 bob clarice") == Foo(a=set(["1", "bob", "clarice"]))
