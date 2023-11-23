# CVOps Python

Command line utility and python library for interacting with CVOps MQTT Broker, validating deployments, and running real-time inference on connected video sources.

## Intallation

Install prequisite binrary packages

```bash
# Linux
sudo apt install libopencv-dev libboost-all-dev libjsoncpp- python3.8-dbg
```

Activate a virtual environment, then run the following:

```bash
pip install -U pip
pip install git+https://github.com/kevin-presalytics/cvops-python.git
```

# Development

To develop, clone the repo and install development mode.

```bash
# clone the repo
git clone https://github.com/kevin-presalytics/cvops-python.git
cd cvops-python

# Create a virtualenv
# Minimum python version is 3.8
python3 -m venv venv

# Activate the virtualenv and upgrade pip
. venv/bin/activate
pip install -U pip

# Install the package with development utilities in editable mode
pip install -e .[dev]
```

### Pre-commit hooks

This library uses client-side pre-commit hooks to ensure code quality of commits.  Install client-side pre-commit hooks with the following commands. 

```bash
. venv/bin/activate
install_hooks
```

### Testing

To run standard unit tests, use the command:

```bash
run_tests
```

