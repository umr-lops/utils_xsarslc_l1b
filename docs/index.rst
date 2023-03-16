############################################################################
xsarslc: functions to compute cross spectra from Sentinel-1 SLC SAR products
############################################################################

**xsarslc** is a library to compute cross spectrum from level 1 SAR SLC products. Objets manipulated are all `xarray`_.

Acquisition modes handled by **xsarslc** are IW, EW and WV.

The input `datatree`_ object can come from any reader library but the original design has been done using `xsar`_


.. jupyter-execute:: examples/intro.py



.. image:: oceanspectrumSAR.png
   :width: 500px
   :height: 400px
   :scale: 110 %
   :alt: real part SAR cross spectrum
   :align: right



Documentation
-------------

Overview
........

    **xsarslc** can compute both intra burst and inter (i.e. overlapping bursts) burst cross spectrum.

    To have comparable cross spectrum among bursts and sub-swaths, we choose to have constant `dk` values,
    it means that the number of pixels used to compute the cross spectrum is not always the same.

    The algorithm is performing 4 different sub-setting in the original complex digital number images:

        1) bursts sub-setting
        2) tiles sub-setting
        3) periodograms sub-setting
        4) looks (in azimuth) sub-setting

    Default configuration is set to:
        * 20x20 km tiles in the bursts. (less in inter burst where we have about 3 km of overlapping).
        * 0% overlapping tiles
        * :math:`2.tau` saved cross spectrum


Algorithm Technical Baseline Document
.....................................

.. note::
    The Algorithm Technical Baseline Document (ATBD) describes implemented processing steps from Sentinel-1 SLC product to cross spectra

* :doc:`ATBD`

Examples
........

.. note::
    here are some examples of usage

* :doc:`examples/xspec_IW_intra_and_inter_burst`
* :doc:`examples/xspec_WV_example`
* :doc:`examples/example_IW_compute_and_correct_from_impulse_response`
* :doc:`examples/example_WV_compute_and_correct_from_impulse_response`
* :doc:`examples/default_impulseResponse_files_IW`


Reference
.........

* :doc:`basic_api`

Get in touch
------------

- Report bugs, suggest features or view the source code `on github`_.

----------------------------------------------

Last documentation build: |today|

.. toctree::
   :maxdepth: 2
   :caption: Home
   :hidden:

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing

.. toctree::
   :maxdepth: 18
   :hidden:
   :caption: Algorithm description

   ATBD


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   examples/xspec_IW_intra_and_inter_burst
   examples/xspec_WV_example
   examples/example_IW_compute_and_correct_from_impulse_response
   examples/example_WV_compute_and_correct_from_impulse_response
   examples/default_impulseResponse_files_IW

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   basic_api

.. _on github: https://github.com/umr-lops/xsar_slc
.. _xsar: https://github.com/umr-lops/xsar
.. _xarray: http://xarray.pydata.org
.. _datatree: https://github.com/xarray-contrib/datatree
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html