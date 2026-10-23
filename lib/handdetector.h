#ifndef HANDDETECTOR_H
#define HANDDETECTOR_H

#include <ncnn/net.h>

#include <opencv2/core/mat.hpp>

#include "mylib/mylib_export.h"

struct Detection {
    cv::Rect_<float> rect;
    float prob;
    uint32_t label = 0;
};

class WYCAI_EXPORT HandDetector {
public:
    HandDetector();
    ~HandDetector();

    bool load_model(const std::string param, const std::string bin);

    std::vector<Detection> detect(const cv::Mat& bgr,
        float conf_thresh = 0.60f,
        float iou_thresh = 0.45f);

    void label(const std::string& img_dir, const std::string& label_dir, const std::string& vis_dir);

private:
    static void nms(std::vector<Detection>& dets, float iou_thresh);
    static float iou(const Detection& a, const Detection& b);

    static void nmsInPlace(std::vector<Detection>& dets, float iou_thresh);

private:
    ncnn::Net net;
    bool loaded = false;
};
#endif // HANDDETECTOR_H
