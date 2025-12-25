cmake_minimum_required(VERSION 3.16)
project(BilliardsTrainer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

# Main executable
add_executable(table_detector
    src/main.cpp
    src/diamond_detection.cpp
    src/felt_detection.cpp
    src/rail_detection.cpp
)

# Set Windows subsystem to prevent console window
if(WIN32)
    # Use WINDOWS subsystem with mainCRTStartup entry point
    # This prevents console window from appearing
    set_target_properties(table_detector PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup"
    )
endif()

# Link OpenCV libraries
target_link_libraries(table_detector ${OpenCV_LIBS})
