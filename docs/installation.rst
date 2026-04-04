Installation
============

Requirements
------------

econirl requires Python 3.10 or later.

Dependencies:

- JAX >= 0.4.30
- Equinox >= 0.11
- Optax >= 0.2
- Gymnasium >= 0.29
- NumPy >= 1.24
- Pandas >= 2.0
- SciPy >= 1.10
- Matplotlib >= 3.7

Installing from PyPI
--------------------

.. code-block:: bash

   pip install econirl

Installing from Source
----------------------

For the latest development version:

.. code-block:: bash

   git clone https://github.com/rawatpranjal/econirl.git
   cd econirl
   pip install -e .

Development Installation
------------------------

To install with development dependencies (testing, linting):

.. code-block:: bash

   pip install -e ".[dev]"

Documentation Dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Verifying Installation
----------------------

After installation, verify everything works:

.. code-block:: python

   import econirl
   print(econirl.__version__)

   # Quick test
   from econirl import RustBusEnvironment
   env = RustBusEnvironment()
   print(env.describe())
