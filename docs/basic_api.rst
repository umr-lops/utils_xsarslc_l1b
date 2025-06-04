#############
reference API
#############

..
    to document functions, add them to __all__ in ../slcl1butils/__init__.py

API = Application Programming Interface


.. automodule:: slcl1butils
    :members: compute,plotting,coloc,legacy_ocean


processing
==========

.. automodule:: slcl1butils.compute.compute_from_l1b
    :members: compute_xs_from_l1b,compute_xs_from_l1b_wv

plotting
========

.. automodule:: slcl1butils.plotting.display_one_spectra
    :members: add_polar_direction_lines


.. automodule:: slcl1butils.plotting.add_azimuth_cutoff_lines_on_polar_spec_fig
    :members: add_azimuth_cutoff_lines

colocation
==========

.. automodule:: slcl1butils.coloc.coloc
    :members: raster_cropping_in_polygon_bounding_box,coloc_tiles_from_l1bgroup_with_raster
