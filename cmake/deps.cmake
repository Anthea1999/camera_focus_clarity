# Project Dependences Configuration

# Find other dependences
set(OpenCV_STATIC OFF CACHE BOOL "Using OpenCV static linking library")
#find_package(OpenCV REQUIRED)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#find_package(Serialport REQUIRED)
find_package(Serialport)
include_directories(${SERIALPORT_INCLUDE_DIRS})

#find_package(Fuzzylite REQUIRED)
find_package(Fuzzylite)
include_directories(${Fuzzylite_INCLUDE_DIRS})

# Include subdirectories
include_directories(${DEPS_PATHS})

# Backup and set build type to release
if(NOT MSVC)
    set(CMAKE_BUILD_TYPE_BAK ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add subdirectory
foreach(DEPS_PATH ${DEPS_PATHS})
    add_subdirectory(${DEPS_PATH})
endforeach()

# Restore origin build type
if(NOT MSVC)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BAK})
endif()

