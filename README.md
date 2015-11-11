# TensorFuse
Common interface for Theano, CGT, and TensorFlow.

## Installation

The recommended way to install this library is `python setup.py develop`.

## Usage

The current API for TensorFuse is geared towards migrating existing Theano-based code to using CGT or TensorFlow. Therefore, if you have some code that uses Theano, you can simply change the import statements to
```python
import tensorfuse as theano
import tensorfuse.tensor as T
```
Then (hopefully) your code can switch between Theano / CGT / TensorFlow easily. The choice is controlled by an environment variable `TENSORFUSE_MODE`, with valid values `theano`, `cgt`, `tensorflow`, or `tf` for abbreviation (if no environment variable is set, then it defaults to cgt). Therefore, to run the code in Theano mode, you can either set the environment variable before running the script, or add the following to the _beginning_ of the file:
```python
import os
os.environ['TENSORFUSE_MODE'] = 'theano'
```
To run the code in TensorFlow mode, do the following:
```python
import os
os.environ['TENSORFUSE_MODE'] = 'tf' # or 'tensorflow'
```

## Running Benchmark

```
python tensorfuse/benchmark/run_benchmark.py tensorfuse/benchmark/simple_ops.py
```
