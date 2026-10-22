Installation
============

Create a conda environment (recommended, since lstchain is distributed via
conda):

.. code-block:: bash

   conda create -n lst_tools -c conda-forge python=3.11 lstchain
   conda activate lst_tools
   pip install lst_tools

For development, install from a clone of the repository with the test and
lint tools:

.. code-block:: bash

   git clone https://github.com/zhipzhang/lst_tools.git
   cd lst_tools
   pip install -e .[dev]

   pytest        # run tests
   ruff check .  # lint
