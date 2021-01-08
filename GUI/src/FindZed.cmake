set(ZEDROOT "/usr/local" CACHE PATH "Root directory of libsl_zed")

FIND_PATH(ZED_INCLUDE_DIR libsl_zed HINTS "${ZED_ROOT}/include")
FIND_LIBRARY(ZED_LIBRARY ZED HINTS "${ZED_ROOT}/bin/x64" "${ZED_ROOT}/lib")

find_package_handle_standard_args(ZED DEFAULT_MSG ZED_LIBRARY ZED_INCLUDE_DIR)

mark_as_advanced(ZED_LIBRARY ZED_INCLUDE_DIR)

