#pragma once
#include "faceInfo.h"

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

class FaceLandmark {
public:
    bool load_model(const std::string& param, const std::string& bin);

    void check106(const cv::Mat& src, FaceInfo& face);

    static constexpr std::array<std::array<float, 2>, 5> STANDARD_106_LANDMARKS = { {
        { 67.0f, 70.0f }, // 左眼
        { 125.0f, 70.0f }, // 右眼
        { 96.0f, 104.0f }, // 鼻尖
        { 77.0f, 134.0f }, // 左嘴角
        { 115.0f, 134.0f } // 右嘴角
    } };

private:
    void check_eye_closed(FaceInfo& face);

    void check_mouth_open(FaceInfo& face);

    void check_head_pose(FaceInfo& face);

private:
    ncnn::Net net;
    const int input_size = 192;

    const float mean[3] = { 127.5f, 127.5f, 127.5f };
    const float norm[3] = { 1.f / 128.f, 1.f / 128.f, 1.f / 128.f };
};
