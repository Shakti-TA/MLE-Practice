def test_package_importable():
    """Test if the key components of the package are importable"""

    try:
        from HousePricePrediction.ingest_data import DataIngestion
    except ImportError as e:
        assert False, f"Could not import DataIngestion: {e}"

    try:
        from HousePricePrediction.train import Data_Train
    except ImportError as e:
        assert False, f"Could not import Data_Train: {e}"

    try:
        from HousePricePrediction.score import Score
    except ImportError as e:
        assert False, f"Could not import Score: {e}"


def test_dependencies_importable():
    """Test if required external dependencies are installed"""
    required_deps = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "scipy",
        "six",
        "black",
        "isort",
        "flake8",
        "pytest",
    ]

    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError as e:
            assert False, f"Dependency '{dep}' not installed: {e}"
