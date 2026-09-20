cmake_minimum_required(VERSION 3.5)

project(qtforcamera VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID)
    set(QT_ANDROID_PACKAGE_SOURCE_DIR "\"android-package-source-directory\": \"${CMAKE_CURRENT_SOURCE_DIR}/android\"," CACHE STRING "" FORCE)
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets)

if(ANDROID)
    find_package(Qt5 REQUIRED COMPONENTS AndroidExtras)
endif()

set(HEADER
    camera.h
    imagesettings.h
    videosettings.h
    productfeatureengine.h
    productdatabase.h
    workthread.h
    imageutils.h
    roiviewfinder.h
    goodsmodel.h
    goodsdelegate.h
)
if(ANDROID)
    list(APPEND HEADER androidUtils/localandroidutils.h)
endif()

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
    goodsmodel.cpp
    goodsdelegate.cpp
)
if(ANDROID)
    list(APPEND HEADER androidUtils/localandroidutils.cpp)
endif()

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

# 针对 MSVC 编译器添加 /utf-8 选项
if(MSVC)
    # 针对 Debug 配置
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/target/bin/Debug)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/target/lib/Debug)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/target/lib/Debug)

    # 针对 Release 配置
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/target/bin/Release)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/target/lib/Release)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/target/lib/Release)

    add_compile_options(/utf-8)
endif()

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(qtforcamera
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
    set_target_properties(qtforcamera PROPERTIES
        QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
    )
else()
    if(ANDROID)
        add_library(qtforcamera SHARED
            ${PROJECT_SOURCES}
        )
        set_target_properties(qtforcamera PROPERTIES
            QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
            # ANDROID_EXTRA_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/libs/libcustom.so"
        )
    else()
        add_executable(qtforcamera
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_include_directories(qtforcamera PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

#包含Android目录
if(ANDROID)
    file(GLOB_RECURSE ANDROID_EXTRA_FILES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/android/*"
    )
    if(ANDROID_EXTRA_FILES)
        target_sources(qtforcamera PRIVATE ${ANDROID_EXTRA_FILES})

        source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/android"
                     PREFIX "android"
                     FILES ${ANDROID_EXTRA_FILES})
    endif()
endif()

#链接
target_link_libraries(qtforcamera PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Multimedia
    Qt${QT_VERSION_MAJOR}::MultimediaWidgets
)

if(ANDROID)
    target_link_libraries(qtforcamera PRIVATE Qt5::AndroidExtras)
endif()

#link opencv
set(OPENCV_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv2")
target_include_directories(qtforcamera PRIVATE "${OPENCV_DLL_DIR}/include")

if(WIN32)
    set(OPENCV_WIN_DLL_DIR "${OPENCV_DLL_DIR}/win")

    target_link_libraries(qtforcamera PRIVATE
        # Debug 模式链接
        debug "${OPENCV_WIN_DLL_DIR}/lib/opencv_core455d.lib"
        debug "${OPENCV_WIN_DLL_DIR}/lib/opencv_imgcodecs455d.lib"
        debug "${OPENCV_WIN_DLL_DIR}/lib/opencv_imgproc455d.lib"

        # Release / RelWithDebInfo 等模式链接
        optimized "${OPENCV_WIN_DLL_DIR}/lib/opencv_core455.lib"
        optimized "${OPENCV_WIN_DLL_DIR}/lib/opencv_imgcodecs455.lib"
        optimized "${OPENCV_WIN_DLL_DIR}/lib/opencv_imgproc455.lib"
    )

    # 3. 自动拷贝对应的 DLL 到输出目录
    add_custom_command(TARGET qtforcamera POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<$<CONFIG:Debug>:${OPENCV_WIN_DLL_DIR}/bin/opencv_core455d.dll>
            $<$<CONFIG:Debug>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgcodecs455d.dll>
            $<$<CONFIG:Debug>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgproc455d.dll>
            $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WIN_DLL_DIR}/bin/opencv_core455.dll>
            $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgcodecs455.dll>
            $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgproc455.dll>
            "$<TARGET_FILE_DIR:qtforcamera>"
        COMMAND_EXPAND_LISTS
    )
elseif(ANDROID)
    set(OPENCV_ANDROID_LIB_DIR "${OPENCV_DLL_DIR}/android")

    add_library(opencv_static STATIC IMPORTED)

    file(GLOB ANDROID_STATIC_LIBS
            "${OPENCV_ANDROID_LIB_DIR}/lib/${ANDROID_ABI}/*.a"
        )

    file(GLOB OPENCV_3RDPARTY_LIBS
            "${OPENCV_ANDROID_LIB_DIR}/3rdparty/lib/${ANDROID_ABI}/*.a"
        )

    target_link_libraries(qtforcamera PRIVATE
        -Wl,--start-group
        ${ANDROID_STATIC_LIBS}
        ${OPENCV_3RDPARTY_LIBS}
        -Wl,--end-group
        log
        z
        android
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
#if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/target/install" CACHE PATH "Default install prefix" FORCE)
#endif()
#include(GNUInstallDirs)
#install(TARGETS qtforcamera
#    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}   # macOS bundle (.app)
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}   # Windows / Linux 可执行文件 (.exe)
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}   # 动态库 (.so / .dylib)
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}   # 静态库 (.a / .lib)
#)

#向下兼容 Qt 5 避免编译时出现 Unknown CMake command "qt_finalize_executable" 的报错
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(qtforcamera)
endif()
