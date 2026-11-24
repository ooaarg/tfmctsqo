include("${CMAKE_CURRENT_LIST_DIR}/../extension_config.cmake")
# SQL regression tests inspect structured diagnostics.
duckdb_extension_load(json)
