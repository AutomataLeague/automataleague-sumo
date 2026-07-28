def test_package_imports_and_has_version():
    import automataleague_sumo

    assert isinstance(automataleague_sumo.__version__, str)
    assert automataleague_sumo.__version__
