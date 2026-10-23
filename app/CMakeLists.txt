if(ANDROID)
    set(QT_ANDROID_PACKAGE_SOURCE_DIR "\"android-package-source-directory\": \"${CMAKE_CURRENT_SOURCE_DIR}/android\"," CACHE STRING "" FORCE)
endif()

set(HEADER
    camera.h
    imagesettings.h
    videosettings.h
    workthread.h
    imageutils.h
    roiviewfinder.h
    goodsmodel.h
    goodsdelegate.h
    customglsurface.h
    glrendererwidget.h
)
if(ANDROID)
    list(APPEND HEADER androidUtils/localandroidutils.h)
endif()

set(SOURCES
    main.cpp
    camera.cpp
    imagesettings.cpp
    videosettings.cpp
    workthread.cpp
    imageutils.cpp
    roiviewfinder.cpp
    goodsmodel.cpp
    goodsdelegate.cpp
    customglsurface.cpp
    glrendererwidget.cpp
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

#增加资源
qt_add_resources(RESOURCES res.qrc)

# 针对 MSVC 编译器添加 /utf-8 选项
if(MSVC)
    add_compile_options(/wd4251)
    add_compile_options(/arch:AVX2)
    add_compile_options(/utf-8)
endif()

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(qtforcamera
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        ${OCR_CORE_SRCS}
        ${RESOURCES}
    )
    set_target_properties(qtforcamera PROPERTIES
        QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
    )
else()
    if(ANDROID)
        add_library(qtforcamera SHARED
            ${PROJECT_SOURCES}
            ${OCR_CORE_SRCS}
            ${RESOURCES}
        )
        set_target_properties(qtforcamera PROPERTIES
            QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
            # ANDROID_EXTRA_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/libs/libcustom.so"
        )
    else()
        add_executable(qtforcamera
            ${PROJECT_SOURCES}
            ${OCR_CORE_SRCS}
            ${RESOURCES}
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
    Qt${QT_VERSION_MAJOR}::OpenGL
)

if(ANDROID)
    find_library(GLESV3_LIB GLESv3 REQUIRED)
elseif(WIN32)
    find_library(OPENGL_LIB opengl32 REQUIRED)
endif()

if(ANDROID)
    target_link_libraries(qtforcamera PRIVATE Qt5::AndroidExtras ${GLESV3_LIB})
elseif(WIN32)
    target_link_libraries(qtforcamera PRIVATE ${OPENGL_LIB})
endif()

target_link_libraries(qtforcamera PRIVATE wyc_ai)

##临时
if(ANDROID)
    set(OPENCV_DLL_DIR "${CMAKE_SOURCE_DIR}/lib/opencv2")
    target_include_directories(qtforcamera PRIVATE "${OPENCV_DLL_DIR}/include")
    file(GLOB ANDROID_STATIC_LIBS "${OPENCV_DLL_DIR}/android/lib/${ANDROID_ABI}/*.a")
    file(GLOB OPENCV_3RDPARTY_LIBS "${OPENCV_DLL_DIR}/android/3rdparty/lib/${ANDROID_ABI}/*.a")
    target_compile_options(qtforcamera PRIVATE -fopenmp)
    target_link_options(qtforcamera PRIVATE -fopenmp -static-openmp)
    target_link_libraries(qtforcamera PRIVATE
        -Wl,--gc-sections -s
        -Wl,--start-group
        ${ANDROID_STATIC_LIBS}
        ${OPENCV_3RDPARTY_LIBS}
        -Wl,--end-group
        log z android
    )
endif()

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
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default install prefix" FORCE)
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
