.. _installing:

************
Installation
************

L1B SAR SLC IFREMER products are netCDF (.nc) files containing groups.
To read netCDF files with groups, a possible python library is `xarray-datatree`.
Installation in a conda_ environment is recommended.


conda install
#############


.. code-block::

    conda create -n l1butilsenv
    conda activate l1butilsenv
    conda install -c conda-forge slcl1butils


pip install
###########


To be up to date with the development team, it's recommended to update the installation using pip:

.. code-block::

    pip install git+https://github.com/umr-lops/utils_xsarslc_l1b.git



dev install
###########

.. code-block::

    git clone https://github.com/umr-lops/utils_xsarslc_l1b.git
    cd utils_xsarslc_l1b
    pip install -e .


.. _conda: https://docs.anaconda.com/anaconda/install/