""" CVOps Tests """
import sys
import pathlib
import unittest
import tests.test_inference
import tests.test_tracker

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
PACKAGE_DIR = ROOT_DIR.joinpath("src", "cvops")

sys.path.insert(0, str(PACKAGE_DIR))

# Add modules to this list to run them during CI/CD pipelines and pre-commit hooks
ALL_TEST_MODULES = [
    tests.test_inference,
    tests.test_tracker,
]


def test_all():
    """ Runs all tests """
    runner = unittest.TextTestRunner()
    failed = False
    for module in ALL_TEST_MODULES:
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = runner.run(suite)
        if not result.wasSuccessful():
            # Required for pre-commit hook to fail
            failed = True
    if failed:
        sys.exit(1)
