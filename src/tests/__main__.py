"""Run all tests."""
import sys
import tests


if __name__ == '__main__':
    try:
        TOOLS_DIR = tests.ROOT_DIR.joinpath("src", "tools")
        sys.path.insert(0, str(TOOLS_DIR))
        import tools  # pylint: disable=import-error
        tools.bootstrap_cmake()
        tests.test_all()
    except Exception as ex:  # pylint: disable=broad-except
        print(ex)
        sys.exit(1)
