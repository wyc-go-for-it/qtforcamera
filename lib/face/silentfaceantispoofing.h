#pragma once

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <string>

// 活体检测结果
struct AntiSpoofingResult {
    bool is_real; // 是否为真人
    float real_score; // 真人概率
    float print_score; // 打印照片攻击概率
    float screen_score; // 屏幕翻拍攻击概率
};

class SilentFaceAntiSpoofing {
public:
    SilentFaceAntiSpoofing();
    ~SilentFaceAntiSpoofing();

    bool load_model(const std::string& param_path, const std::string& bin_path);

    // 输入原图和裁剪好的人脸 BBox
    AntiSpoofingResult detect(const cv::Mat& img, const cv::Rect& face_rect);

private:
    ncnn::Net net;
    const int input_size = 80; // MiniFASNet 通常输入 80x80

    // 预处理参数 (大部分 MiniFASNet 采用 /255.0 的归一化)
    const float mean[3] = { 0.f, 0.f, 0.f };
    const float norm[3] = { 1.f / 255.f, 1.f / 255.f, 1.f / 255.f };
};
