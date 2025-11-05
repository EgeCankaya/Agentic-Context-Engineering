def foo(x: str) -> str:
    return x


def test_foo():
    assert foo("foo") == "foo"
