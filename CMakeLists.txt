cmake_minimum_required(VERSION 3.5)

project(qtforcamera VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets OpenGL)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets OpenGL)

if(ANDROID)
    find_package(Qt5 REQUIRED COMPONENTS AndroidExtras)
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(WIN32)
    set(TARGET_OUT_DIR "${CMAKE_SOURCE_DIR}/target/win")
elseif(ANDROID)
    set(TARGET_OUT_DIR "${CMAKE_SOURCE_DIR}/target/android")
endif()

if(WIN32)
    # 针对 Debug 配置
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUT_DIR}/bin/Debug)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUT_DIR}/lib/Debug)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUT_DIR}/lib/Debug)

    # 针对 Release 配置
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUT_DIR}/bin/Release)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUT_DIR}/lib/Release)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUT_DIR}/lib/Release)
endif()

add_subdirectory(lib)
add_subdirectory(app)
