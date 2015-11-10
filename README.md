# cgtcompat
Common interface for CGT and Theano

## Installation

The recommended way to install this library is `python setup.py develop`.

## Usage

The current API for `cgtcompat` is geared towards migrating existing Theano-based code to using cgt. Therefore, if you have some code that uses Theano, you can simply change the import statements to
```python
import cgtcompat as theano
import cgtcompat.tensor as T
```
Then (hopefully) your code can switch between Theano and cgt easily. The choice is controlled by an environment variable `CGT_COMPAT_MODE`, with valid values `theano` or `cgt` (if no environment variable is set, then it defaults to cgt). Therefore, to run the code in Theano mode, you can either set the environment variable before running the script, or add the following to the _beginning_ of the file:
```python
import os
os.environ['CGT_COMPAT_MODE'] = 'theano' # or 'cgt'
```
