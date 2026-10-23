#ifndef YOLO11_NCNN_H
#define YOLO11_NCNN_H

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "mylib/mylib_export.h"

struct Object {
    cv::Rect_<float> rect; // 目标框 [x, y, width, height]
    int label; // 类别 ID
    float prob; // 置信度
};

struct LetterboxParams {
    float scale;
    int pad_w;
    int pad_h;
};

class WYCAI_EXPORT Yolo11Ncnn {
public:
    Yolo11Ncnn();
    ~Yolo11Ncnn();

    bool load_model(const std::string& param_path,
        const std::string& bin_path,
        int target_size = 640,
        bool use_gpu = false);

    bool detect(const cv::Mat& bgr,
        std::vector<Object>& objects,
        float conf_thresh = 0.25f,
        float nms_thresh = 0.45f);

private:
    ncnn::Net net;
    int target_size;
    bool is_loaded;

    ncnn::Mat preprocess_letterbox(const cv::Mat& bgr, LetterboxParams& params);
    void nms_sorted_bboxes(const std::vector<Object>& objects,
        std::vector<Object>& picked,
        float nms_thresh);
    static inline float intersection_area(const Object& a, const Object& b);

    bool detect_modelA(const cv::Mat& bgr,
        std::vector<Object>& objects,
        float conf_thresh,
        float nms_thresh);

    bool detect_modelB(const cv::Mat& bgr,
        std::vector<Object>& objects,
        float conf_thresh,
        float nms_thresh);
};

#endif // YOLO11_NCNN_H
