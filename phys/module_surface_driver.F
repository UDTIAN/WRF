!WRF:MEDIATION_LAYER:PHYSICS
!
MODULE module_surface_driver
  real, private, parameter :: QVAPOR_MIN = 0.0, SMOIS_MIN = 0.0, SMOIS_MAX = 1.0
  real, private, parameter :: SOIL_OVERRIDE_CONST = 0.30
  real, private, parameter :: LAT_MIN_SOIL = 32.0, LAT_MAX_SOIL = 40.0
  real, private, parameter :: LON_MIN_SOIL = -110.0, LON_MAX_SOIL = -100.0
CONTAINS