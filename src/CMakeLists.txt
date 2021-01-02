# CMakeList.txt : CMake project for src, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 20)	
set(CMAKE_CXX_FLAGS "-O3 -Wall -ffast-math")

find_package(PythonLibs 3.8 REQUIRED)
find_package(PythonInterp 3.8 REQUIRED)


include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("D:/vs_dev_lib/pybind11-2.6.1/include")
include_directories("${CMAKE_SOURCE_DIR}/include")

# targets
add_library(
	core
	SHARED
	"pybind_export.cpp"
	)

set_target_properties(core PROPERTIES PREFIX "")
set_target_properties(core PROPERTIES OUTPUT_NAME "_core")

if (WIN32)
	set_target_properties(core PROPERTIES SUFFIX ".pyd")
else()
	set_target_properties(core PROPERTIES SUFFIX ".so")
endif()

target_link_libraries(core "${PYTHON_LIBRARIES}")

#
add_executable(
	main0
	"main0.cpp"
)