Datasets
========

.. module:: econirl.datasets

Built-in datasets for replication and teaching purposes.

load_rust_bus
-------------

.. autofunction:: econirl.datasets.load_rust_bus

   Load the Rust (1987) bus engine replacement dataset.

   **Example:**

   .. code-block:: python

      from econirl.datasets import load_rust_bus

      # Load full dataset as DataFrame
      df = load_rust_bus()
      print(f"Shape: {df.shape}")
      print(f"Columns: {list(df.columns)}")

      # Load specific group
      df_group1 = load_rust_bus(group=1)

      # Load as Panel for estimation
      panel = load_rust_bus(as_panel=True)
      print(f"Individuals: {panel.num_individuals}")

   **Dataset Structure:**

   The dataset contains monthly observations with the following columns:

   - ``bus_id``: Unique bus identifier
   - ``period``: Month number (1-indexed)
   - ``mileage``: Odometer reading (thousands of miles)
   - ``mileage_bin``: Discretized mileage state (0-89)
   - ``replaced``: 1 if engine was replaced, 0 otherwise
   - ``group``: Bus group (1-8)

   **Groups:**

   Groups 1-4 are GMC buses (Rust's main focus). Groups 5-8 are from different
   manufacturers with different characteristics.

get_rust_bus_info
-----------------

.. autofunction:: econirl.datasets.rust_bus.get_rust_bus_info

   Get metadata about the Rust bus dataset.

   **Example:**

   .. code-block:: python

      from econirl.datasets.rust_bus import get_rust_bus_info

      info = get_rust_bus_info()
      print(f"Observations: {info['n_observations']}")
      print(f"Buses: {info['n_buses']}")
      print(f"Replacement rate: {info['replacement_rate']:.2%}")

Adding Custom Datasets
----------------------

To add your own datasets to econirl:

1. Create a module in ``econirl/datasets/``
2. Implement a loader function returning DataFrame or Panel
3. Export from ``econirl/datasets/__init__.py``

**Example:**

.. code-block:: python

   # econirl/datasets/my_data.py

   import pandas as pd
   from pathlib import Path

   def load_my_data():
       \"\"\"Load my custom dataset.\"\"\"
       data_path = Path(__file__).parent / "my_data.csv"
       return pd.read_csv(data_path)

   # econirl/datasets/__init__.py
   from econirl.datasets.my_data import load_my_data
