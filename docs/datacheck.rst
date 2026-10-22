Loading DataCheck statistics
============================

The nightly LST-1 DataCheck HDF5 files can be loaded and reduced to
run-wise statistics with :class:`~lst_tools.datacheck.DataCheckTables`:

.. code-block:: python

   from lst_tools.datacheck import DataCheckTables

   tables = DataCheckTables.from_files(["/path/to/DL1_datacheck_20240101.h5"])
   statistics = tables.statistics          # one row per run

   # keep a subset of runs and write a new datacheck file
   tables.select_runs([14083, 14084]).save_to_h5file("selected.h5", overwrite=True)

Quality cuts are applied with :class:`~lst_tools.datacheck.DataFilter`:

.. code-block:: python

   from lst_tools.datacheck import DataFilter

   data_filter = DataFilter(
       source_ra=83.632,
       source_dec=22.0145,
       min_angle_to_source=0.3,
       max_angle_to_source=0.5,
       max_pointing_dec_std=0.01,
   )
   after_basic = data_filter.apply_basic_cuts(statistics)
   selected = data_filter.apply_advanced_cuts(after_basic)

Zenith binning helpers (:func:`~lst_tools.datacheck.assign_zenith_bins`,
:func:`~lst_tools.datacheck.zenith_bin_labels`,
:func:`~lst_tools.datacheck.validate_zenith_bin_edges`) group the selected
runs into bins. Bins include their lower edge and exclude their upper edge;
the last bin also includes its upper edge.
