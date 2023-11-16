""" CVOps Tests """
import sys
import pathlib
import unittest
import tests.test_inference

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
PACKAGE_DIR = ROOT_DIR.joinpath("src", "cvops")

sys.path.insert(0, str(PACKAGE_DIR))

def test_suite():
    """ Collects tests into test suite """
    suite = unittest.TestSuite()
    suite.addTest(tests.test_inference.TestInference("test_inference"))
    return suite

def run_tests():
    """ Runs all tests """
    runner = unittest.TextTestRunner()
    suite = test_suite()
    runner.run(suite)