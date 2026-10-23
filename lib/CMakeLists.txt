if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.5)
    project(wyc_ai CXX)
endif()

option(ENABLE_LOG "Enable runtime logs" ON)

set(hasStatic ON CACHE BOOL "Enable static build" FORCE)

set(HEADER
    productfeatureengine.h
    productdatabase.h
    producttextcleaner.h
    yoloncnn.h
    bboxu2netp.h
    isnetsegmenter.h
    litesaliencynet.h
    handdetector.h
)

set(SOURCES
    productfeatureengine.cpp
    productdatabase.cpp
    producttextcleaner.cpp
    yoloncnn.cpp
    bboxu2netp.cpp
    isnetsegmenter.cpp
    litesaliencynet.cpp
    handdetector.cpp
)

set(PROJECT_SOURCES
    ${SOURCES}
    ${HEADER}
)

#目标输出配置
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(WIN32)
        set(TARGET_OUT_DIR "${CMAKE_SOURCE_DIR}/target/win")
    elseif(ANDROID)
        set(TARGET_OUT_DIR "${CMAKE_SOURCE_DIR}/target/android")
    endif()

    # 针对 Debug 配置
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUT_DIR}/bin/Debug)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUT_DIR}/lib/Debug)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUT_DIR}/lib/Debug)

    # 针对 Release 配置
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUT_DIR}/bin/Release)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUT_DIR}/lib/Release)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUT_DIR}/lib/Release)
endif()
# 针对 MSVC 编译器添加 /utf-8 选项
if(MSVC)
    add_compile_options(/wd4251)
    add_compile_options(/arch:AVX2)

    add_compile_options(/utf-8)
endif()
#end 目标输出配置

#ocr
set(OCR_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ocr_src")

set(OCR_INCLUDE "${OCR_DLL_DIR}/include")
include_directories(
    ${OCR_INCLUDE}
)

file(GLOB OCR_CORE_SRCS ${OCR_DLL_DIR}/*.cpp)
file(GLOB OCR_CORE_HEADER ${OCR_INCLUDE}/*.h)

set(OCR_SOURCES
    ${OCR_CORE_SRCS}
    ${OCR_CORE_HEADER}
)
#end ocr

set(TARGET_NAME wyc_ai)

add_library(${TARGET_NAME} SHARED
    ${PROJECT_SOURCES}
    ${OCR_SOURCES}
 )

set(OPENCV_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv2")
if(WIN32)
    if(hasStatic)
        set(NCNN_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/win/static")
    else()
        set(NCNN_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/win")
    endif()

elseif(ANDROID)
     set(NCNN_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/android")
endif()

target_compile_definitions(${TARGET_NAME}
    PUBLIC WYCAI_EXPORT_DLL
)


if(ENABLE_LOG)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LOG)
endif()
target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LOG)

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>

        $<BUILD_INTERFACE:${OCR_DLL_DIR}/include>
        $<BUILD_INTERFACE:${OPENCV_DLL_DIR}/include>
        $<BUILD_INTERFACE:${NCNN_DLL_DIR}/include>

        $<INSTALL_INTERFACE:include>
)

#link ncnn
if(WIN32)
    add_definitions(-DNOMINMAX)

    #搜出所有 .lib 文件
    file(GLOB ALL_LIBS "${NCNN_DLL_DIR}/lib/*.lib")

    #从中过滤出 Debug 库（带 d.lib 结尾的）
    set(NCNN_DEBUG_LIBS ${ALL_LIBS})
    list(FILTER NCNN_DEBUG_LIBS INCLUDE REGEX ".*d\\.lib$")

    #从中过滤出 Release 库（排除带 d.lib 结尾的）
    set(NCNN_RELEASE_LIBS ${ALL_LIBS})
    list(FILTER NCNN_RELEASE_LIBS EXCLUDE REGEX ".*d\\.lib$")

    target_link_libraries(${TARGET_NAME} PUBLIC
        $<$<CONFIG:Debug>:${NCNN_DEBUG_LIBS}>
        $<$<CONFIG:Release>:${NCNN_RELEASE_LIBS}>
    )

    if(hasStatic)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove -f
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/ncnn.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/ncnnd.dll"
            COMMENT "Remove ncnn from output directory"
        )
    else()
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<$<CONFIG:Debug>:${NCNN_DLL_DIR}/bin/ncnnd.dll>
                $<$<NOT:$<CONFIG:Debug>>:${NCNN_DLL_DIR}/bin/ncnn.dll>
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMAND_EXPAND_LISTS
        )
    endif()

elseif(ANDROID)
    set(NCNN_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/android/${ANDROID_ABI}/lib/*.a")

    target_compile_options(${TARGET_NAME} PRIVATE -fopenmp)
    target_link_options(${TARGET_NAME} PRIVATE -fopenmp -static-openmp)

    target_link_libraries(${TARGET_NAME} PRIVATE
        -Wl,--gc-sections -s
        ${NCNN_LIBS}
        log
    )
endif()
#end link ncnn

#link opencv
if(WIN32)
    if(hasStatic)
        set(OPENCV_WIN_DLL_DIR "${OPENCV_DLL_DIR}/win/static")
    else()
        set(OPENCV_WIN_DLL_DIR "${OPENCV_DLL_DIR}/win")
    endif()

    #搜出所有 .lib 文件
    file(GLOB ALL_LIBS "${OPENCV_WIN_DLL_DIR}/lib/*.lib")

    #从中过滤出 Debug 库（带 d.lib 结尾的）
    set(OPENCV_DEBUG_LIBS ${ALL_LIBS})
    list(FILTER OPENCV_DEBUG_LIBS INCLUDE REGEX ".*d\\.lib$")

    #从中过滤出 Release 库（排除带 d.lib 结尾的）
    set(OPENCV_RELEASE_LIBS ${ALL_LIBS})
    list(FILTER OPENCV_RELEASE_LIBS EXCLUDE REGEX ".*d\\.lib$")

    target_link_libraries(${TARGET_NAME} PUBLIC
        $<$<CONFIG:Debug>:${OPENCV_DEBUG_LIBS}>
        $<$<CONFIG:Release>:${OPENCV_RELEASE_LIBS}>
    )

    if(NOT hasStatic)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<$<CONFIG:Debug>:${OPENCV_WIN_DLL_DIR}/bin/opencv_core455d.dll>
                $<$<CONFIG:Debug>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgcodecs455d.dll>
                $<$<CONFIG:Debug>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgproc455d.dll>
                $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WIN_DLL_DIR}/bin/opencv_core455.dll>
                $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgcodecs455.dll>
                $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WIN_DLL_DIR}/bin/opencv_imgproc455.dll>
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMAND_EXPAND_LISTS
        )
    else()
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove -f
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/opencv_core455d.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/opencv_imgcodecs455d.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/opencv_imgproc455d.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/opencv_core455.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/opencv_imgcodecs455.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/opencv_imgproc455.dll"
            COMMENT "Remove OpenCV DLLs from output directory"
        )
    endif()

elseif(ANDROID)
    set(OPENCV_ANDROID_LIB_DIR "${OPENCV_DLL_DIR}/android")

    add_library(opencv_static STATIC IMPORTED)

    file(GLOB ANDROID_STATIC_LIBS
            "${OPENCV_ANDROID_LIB_DIR}/lib/${ANDROID_ABI}/*.a"
        )

    file(GLOB OPENCV_3RDPARTY_LIBS
            "${OPENCV_ANDROID_LIB_DIR}/3rdparty/lib/${ANDROID_ABI}/*.a"
        )

    target_link_libraries(${TARGET_NAME} PRIVATE
        -Wl,--gc-sections -s
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
