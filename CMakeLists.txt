cmake_minimum_required(VERSION 3.5)

project(qtforcamera VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets)

set(HEADER
    camera.h
    imagesettings.h
    videosettings.h
    productfeatureengine.h
    productdatabase.h
    workthread.h
    imageutils.h
    roiviewfinder.h
)

set(SOURCES
    main.cpp
    camera.cpp
    imagesettings.cpp
    videosettings.cpp
    productfeatureengine.cpp
    productdatabase.cpp
    workthread.cpp
    imageutils.cpp
    roiviewfinder.cpp
)

set(UI
    camera.ui
    imagesettings.ui
    videosettings.ui
)

set(PROJECT_SOURCES
    ${SOURCES}
    ${HEADER}
    ${UI}
)

# 针对 Debug 配置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/target/bin/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/target/lib/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/target/lib/Debug)

# 针对 Release 配置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/target/bin/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/target/lib/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/target/lib/Release)

# 针对 MSVC 编译器添加 /utf-8 选项
if(MSVC)
    add_compile_options(/utf-8)
endif()

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(qtforcamera
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET qtforcamera APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(qtforcamera SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(qtforcamera
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(qtforcamera PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Multimedia
    Qt${QT_VERSION_MAJOR}::MultimediaWidgets
)

#link opencv
if(WIN32)
    add_library(OpencvLibrary SHARED IMPORTED)
    set(OPENCV_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv")
    set_target_properties(OpencvLibrary PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCV_DLL_DIR}/include"
    )
    set_target_properties(OpencvLibrary PROPERTIES
        # --- Debug 配置 ---
        IMPORTED_IMPLIB_DEBUG     "${OPENCV_DLL_DIR}/lib/opencv_world452d.lib"
        IMPORTED_LOCATION_DEBUG   "${OPENCV_DLL_DIR}/bin/opencv_world452d.dll"

        # --- Release 配置 ---
        IMPORTED_IMPLIB_RELEASE   "${OPENCV_DLL_DIR}/lib/opencv_world452.lib"
        IMPORTED_LOCATION_RELEASE "${OPENCV_DLL_DIR}/bin/opencv_world452.dll"
    )

    target_link_libraries(qtforcamera PRIVATE OpencvLibrary)
    set(OPENCV_DLL_NAME "opencv_world452.dll")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OPENCV_DLL_NAME "opencv_world452d.dll")
    endif()
    add_custom_command(TARGET qtforcamera POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL_DIR}/bin/${OPENCV_DLL_NAME}"
            "$<TARGET_FILE_DIR:qtforcamera>"
    )
endif()
#end link opencv

#自动打包
if(WIN32)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${Qt${QT_VERSION_MAJOR}Core_DIR}/../../../bin")
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET qtforcamera POST_BUILD
            COMMAND ${WINDEPLOYQT_EXECUTABLE}
                --no-compiler-runtime       # 避免覆盖 VC++ 运行库（可选）
                --no-svg                    # 如果没用到 SVG 插件可以排除，精简体积（可选）
                --verbose 0                 # 减少控制台输出日志
                "$<TARGET_FILE:qtforcamera>"
            COMMENT "Running windeployqt to copy required Qt DLLs..."
        )
    else()
        message(WARNING "windeployqt.exe not found! Qt DLLs won't be copied automatically.")
    endif()
endif()

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.qtforcamera)
endif()
set_target_properties(qtforcamera PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

#安装
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/target/install" CACHE PATH "Default install prefix" FORCE)
endif()
include(GNUInstallDirs)
install(TARGETS qtforcamera
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}   # macOS bundle (.app)
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}   # Windows / Linux 可执行文件 (.exe)
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}   # 动态库 (.so / .dylib)
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}   # 静态库 (.a / .lib)
)

#向下兼容 Qt 5 避免编译时出现 Unknown CMake command "qt_finalize_executable" 的报错
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(qtforcamera)
endif()
