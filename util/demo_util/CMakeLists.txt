cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME demo_util)  # Set project name manually
#get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)  # Set project name with dir name
set(PROJECT_LANGUAGE C CXX)
set(PROJECT_DEPS demo_lib robot_eye ${OpenCV_LIBS} pthread v4l2)

# Compile setting
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 14)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set project
project(${PROJECT_NAME} ${PROJECT_LANGUAGE})

# Add definition
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
    #add_definitions(-D_GLIBCXX_DEBUG)
endif()

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set file list
file(GLOB PROJECT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

# Build utility
add_executable(${PROJECT_NAME} ${PROJECT_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
    )
target_link_libraries(${PROJECT_NAME} ${PROJECT_DEPS})

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    )
