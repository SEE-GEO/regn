Data
====

The input data that we are using is the same that is used to build the
retrieval databases for the current version of the GPROF algorithm. For
each sensor in the GPM constellation a separate retrieval database is
constructed.

We currently have two different datasets to work with: The GMI dataset and the
MHS dataset. The GMI dataset consists of co-located observations from the
Dual-Frequency Precipitation Radar (`DPR
<https://pmm.nasa.gov/GPM/flight-project/DPR>`_) and the GPM Microwave Imager
(`GMI <https://pmm.nasa.gov/gpm/flight-project/gmi>`_) both flown on the GPM
Core Observatory satellite. The MHS dataset consists of simulated observations
from the Microwave humidity sounder (`MHS <https://wdc.dlr.de/sensors/mhs/>`_)
onboard the MetOp-A satellites based on DPR measurements.

Documentation
^^^^^^^^^^^^^

Although the data that we are using does not stem from an official product, it
the documentation for the GPM data products can be helpful. In particular the
algorithm theoretical basis document (ATBD) for the `GPROF algorithm
<https://pmm.nasa.gov/sites/default/files/document_files/GPROF_ATBD_GPM_Aug1_2014.pdf>`_
and the file format specification for all `GPM products
<ftp://gpmweb2.pps.eosdis.nasa.gov/pub/GPMfilespec/filespec.GPM.pdf>`_ can be
helpful.

Training data
^^^^^^^^^^^^^



The input data consists of the observed brightness temperatures of each sensor
and three additional input variables. The ancillary information consists of
the surface type, the total column water vapor and the two-meter temperature.
In the GPROF algorithm the ancillary input data is used to preselect a retrieval
database. This means that, depending of the values of these three variables, a
different retrieval database with different a priori statistics is used.

Training and test data is available in the :code:`REGN` project directory on
Dendrite in the :code:`data` subfolder. The data is stored as NetCDF4 file and
contains the following variables:

+--------------------------------+----------------------------------------------+------------+
| Name                           |  Description                                 |  Type      |
+================================+==============================================+============+
| :code:`year`                   | the year                                     | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`month`                  | the month                                    | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`day`                    | the day                                      | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`hour`                   | the hour                                     | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`minute`                 | minute                                       | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`second`                 | second                                       | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`latitude`               | Latitude of the pixel                        | :code:`f4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`longitude`              | Longitude of the pixel                       | :code:`f4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`brightness_temperature` | The brightness temperatures for each channel | :code:`f4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`surface_type`           | Surface type as class                        | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`tcwv`                   | Total column water vapor class               | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`t2m`                    | Two-meter temperature class                  | :code:`i4` |
+--------------------------------+----------------------------------------------+------------+
| :code:`surface_precipitation`  | Surface precipitation                        | :code:`f4` |
+--------------------------------+----------------------------------------------+------------+

Note that :code:`tcwv`, :code:`t2m` and :code:`surface_type` are given not as continuous 
variables but instead as discrete classes.

:code:`surface_precipitation` is our retrieval target.

Surface types
^^^^^^^^^^^^^

Surface types are encoded as follows:

+-----+---------------------------+
|Code | Class                     |
+=====+===========================+
|1    | Ocean                     |
+-----+---------------------------+
|2    | Sea-Ice                   |
+-----+---------------------------+
|3-7  | Decreasing vegetation     |
+-----+---------------------------+
|8-11 | Decreasing snow cover     |
+-----+---------------------------+
|12   | Standing Water            |
+-----+---------------------------+
|13   | Land/ocean or water Coast |
+-----+---------------------------+
|14   | Sea-ice edge              |
+-----+---------------------------+
|-99  |  Missing value            |
+-----+---------------------------+
