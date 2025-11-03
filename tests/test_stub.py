"""Stub test file to establish test infrastructure."""

import lionherd


def test_package_import():
    """Test that the lionherd package can be imported."""
    assert lionherd is not None


def test_version():
    """Test that the package version is defined."""
    assert hasattr(lionherd, "__version__")
    assert isinstance(lionherd.__version__, str)
    assert lionherd.__version__ == "1.0.0-alpha"


def test_stub():
    """Basic stub test that always passes."""
    assert True
