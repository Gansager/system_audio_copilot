def test_version_import():
    from version import __version__
    assert isinstance(__version__, str) and len(__version__) > 0


