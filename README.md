## Ember
A toolkit for experimenting with novel heuristic algorithms to minor embed QUBO graphs for quantum annealing

### Setup Instructions

1. Install [OR-Tools C++](https://developers.google.com/optimization/install/cpp) for your operating system. Then export the installation directory as:

```bash
export ORTOOLS_DIR=/path/to/ortools
```

**Note**: It is possible to skip this step, however, constraint loading will fallback to the Python API which has a large performance penalty.

2. Make sure you have python>=3.7. If you have pip version >=19.0, then simply invoke a PEP-517 install by running:

```
pip install ./ember
```

3. (Alternative) You can also install with [poetry](https://python-poetry.org/).

```
poetry install
```

4. Run the example script

```
python3 driver.py
```
