// include/mylib/mylib_export.h
#pragma once

#ifdef WYCAI_STATIC
#  define WYCAI_EXPORT
#elif defined(_WIN32) || defined(__CYGWIN__)
#  ifdef WYCAI_EXPORT_DLL
#    define WYCAI_EXPORT __declspec(dllexport)
#  else
#    define WYCAI_EXPORT __declspec(dllimport)
#  endif
#else
#  ifdef WYCAI_EXPORT_DLL
#    define WYCAI_EXPORT __attribute__((visibility("default")))
#  else
#    define WYCAI_EXPORT
#  endif
#endif
