#ifndef FACEENGINE_H
#define FACEENGINE_H

#pragma once

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "mylib/mylib_export.h"

#include "facelandmark.h"

#include <cstddef>

/**
 * @brief The FaceEngine class
 *使用 insightface 模型
 */
class WYCAI_EXPORT FaceEngine {
public:
    FaceEngine();
    ~FaceEngine();

    bool loadDetModel(const std::string& param, const std::string& bin);
    bool loadRecModel(const std::string& param, const std::string& bin);
    bool loadLmModel(const std::string& param, const std::string& bin);

    std::vector<FaceInfo> detectFace(const cv::Mat& img);
    void extract_feature(FaceInfo& face);

    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

    static constexpr size_t featureDim = 512;

private:
    void generateProposals(const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob,
        const ncnn::Mat& kps_blob, int stride, float threshold,
        int img_w, int img_h, int pad_w, int pad_h, float scale,
        std::vector<FaceInfo>& faces);

private:
    static constexpr std::array<std::array<float, 2>, 5> STANDARD_LANDMARKS = { { { 38.2946f, 51.6963f },
        { 73.5318f, 51.5014f },
        { 56.0252f, 71.7366f },
        { 41.5493f, 92.3655f },
        { 70.7299f, 92.2041f } } };

    const std::string det_input_name = "in0";
    const std::vector<std::string> det_output_names = {
        "out0", "out1", "out2",
        "out3", "out4", "out5",
        "out6", "out7", "out8"
    };
    const int det_input_size = 320;
    const std::string rec_input_name = "in0";
    const std::string rec_output_name = "out0";
    const int rec_input_size = 112;

    ncnn::Net det_net;
    ncnn::Net rec_net;
    bool loaded = false;

    FaceLandmark lm_engine;
};
#endif // FACEENGINE_H
