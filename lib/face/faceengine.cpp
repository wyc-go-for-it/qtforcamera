#include "FaceEngine.h"
#include "ncnn/cpu.h"
#include <algorithm>
#include <cmath>

#ifdef ENABLE_LOG
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "HandDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...)                 \
    do {                          \
        std::printf(__VA_ARGS__); \
        fflush(stdout);           \
    } while (0)
#define LOGE(...)                          \
    do {                                   \
        std::fprintf(stderr, __VA_ARGS__); \
        fflush(stderr);                    \
    } while (0)
#endif
#else
#define LOGI(...) ((void)0)
#define LOGE(...) ((void)0)
#endif

static float iou(const cv::Rect_<float>& a, const cv::Rect_<float>& b)
{
    float xx1 = std::max(a.x, b.x);
    float yy1 = std::max(a.y, b.y);
    float xx2 = std::min(a.x + a.width, b.x + b.width);
    float yy2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    if (inter <= 0)
        return 0.f;
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<FaceInfo> nms_faces(std::vector<FaceInfo>& faces, float iou_threshold)
{
    if (faces.empty())
        return {};
    std::sort(faces.begin(), faces.end(), [](const FaceInfo& a, const FaceInfo& b) {
        return a.score > b.score;
    });
    std::vector<int> keep;
    std::vector<bool> removed(faces.size(), false);
    for (size_t i = 0; i < faces.size(); ++i) {
        if (removed[i])
            continue;
        keep.push_back((int)i);
        for (size_t j = i + 1; j < faces.size(); ++j) {
            if (removed[j])
                continue;
            if (iou(faces[i].rect, faces[j].rect) > iou_threshold) {
                removed[j] = true;
            }
        }
    }
    std::vector<FaceInfo> out;
    for (int idx : keep)
        out.push_back(faces[idx]);
    return out;
}

FaceEngine::FaceEngine() { }
FaceEngine::~FaceEngine() { }

bool FaceEngine::loadDetModel(const std::string& param, const std::string& bin)
{
#if defined(__ANDROID__)
    det_net.opt.num_threads = ncnn::get_big_cpu_count();

    det_net.opt.use_fp16_packed = true;
    det_net.opt.use_fp16_storage = true;
    det_net.opt.use_fp16_arithmetic = true;
#else
    det_net.opt.num_threads = 2;
    det_net.opt.use_fp16_packed = false;
    det_net.opt.use_fp16_storage = false;
    det_net.opt.use_fp16_arithmetic = false;
#endif

    det_net.opt.blob_allocator = 0;
    det_net.opt.workspace_allocator = 0;
    det_net.opt.openmp_blocktime = 0;

    det_net.opt.use_packing_layout = true;
    det_net.opt.use_winograd_convolution = true;
    det_net.opt.use_sgemm_convolution = true;

    det_net.opt.lightmode = true;

    if (det_net.load_param(param.c_str()) != 0) {
        LOGE("[FaceEngine] load_param failed: %s\n", param.c_str());
        return false;
    }
    if (det_net.load_model(bin.c_str()) != 0) {
        LOGE("[FaceEngine] load_model failed: %s\n", bin.c_str());
        return false;
    }
    return true;
}

bool FaceEngine::loadRecModel(const std::string& param, const std::string& bin)
{

#if defined(__ANDROID__)
    rec_net.opt.num_threads = ncnn::get_big_cpu_count();

    rec_net.opt.use_fp16_packed = true;
    rec_net.opt.use_fp16_storage = true;
    rec_net.opt.use_fp16_arithmetic = true;
#else
    rec_net.opt.num_threads = 2;
    rec_net.opt.use_fp16_packed = false;
    rec_net.opt.use_fp16_storage = false;
    rec_net.opt.use_fp16_arithmetic = false;
#endif

    rec_net.opt.blob_allocator = 0;
    rec_net.opt.workspace_allocator = 0;
    rec_net.opt.openmp_blocktime = 0;

    rec_net.opt.use_packing_layout = true;
    rec_net.opt.use_winograd_convolution = true;
    rec_net.opt.use_sgemm_convolution = true;

    rec_net.opt.lightmode = true;

    if (rec_net.load_param(param.c_str()) != 0) {
        LOGE("[FaceEngine] load_param failed: %s\n", param.c_str());
        return false;
    }
    if (rec_net.load_model(bin.c_str()) != 0) {
        LOGE("[FaceEngine] load_model failed: %s\n", bin.c_str());
        return false;
    }

    return true;
}

bool FaceEngine::loadLmModel(const std::string& param, const std::string& bin)
{
    return lm_engine.load_model(param, bin);
}

bool FaceEngine::loadSilentFaceAntiSpoofingModel(const std::string& param, const std::string& bin)
{
    return m_silentFaceAntiSpoofing.load_model(param, bin);
}

std::vector<FaceInfo> FaceEngine::detectFace(const cv::Mat& frame)
{
    std::vector<FaceInfo> faces;
    if (frame.empty())
        return faces;

    cv::Mat img = frame;
    if (frame.channels() == 4)
        cv::cvtColor(frame, img, cv::COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cv::cvtColor(frame, img, cv::COLOR_GRAY2BGR);

    auto t0 = std::chrono::high_resolution_clock::now();

    //  Letterbox
    int img_w = img.cols, img_h = img.rows;
    float scale = std::min((float)det_input_size / img_w, (float)det_input_size / img_h);
    int new_w = (int)(img_w * scale);
    int new_h = (int)(img_h * scale);
    int pad_w = (det_input_size - new_w) / 2;
    int pad_h = (det_input_size - new_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(det_input_size, det_input_size, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // 推理
    ncnn::Mat in = ncnn::Mat::from_pixels(
        padded.data, ncnn::Mat::PIXEL_BGR,
        det_input_size, det_input_size);

    const float mean[3] = { 127.5f, 127.5f, 127.5f };
    const float norm[3] = { 1.f / 128.f, 1.f / 128.f, 1.f / 128.f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = det_net.create_extractor();
    ex.input(det_input_name.c_str(), in);

    std::vector<ncnn::Mat> outs(9);
    for (int i = 0; i < 9; ++i) {
        if (ex.extract(det_output_names[i].c_str(), outs[i]) != 0)
            return faces;
    }

    // 后处理（带 pad 还原）
    float threshold = 0.7f;
    generateProposals(outs[0], outs[3], outs[6], 8, threshold, img_w, img_h, pad_w, pad_h, scale, faces);
    generateProposals(outs[1], outs[4], outs[7], 16, threshold, img_w, img_h, pad_w, pad_h, scale, faces);
    generateProposals(outs[2], outs[5], outs[8], 32, threshold, img_w, img_h, pad_w, pad_h, scale, faces);

    std::vector<FaceInfo> final_faces = nms_faces(faces, 0.25f);

    auto t1 = std::chrono::high_resolution_clock::now();

    // 对齐
    for (auto& f : final_faces) {
        auto spoof_res = m_silentFaceAntiSpoofing.detect(img, f.rect);
        if (spoof_res.is_real) {
            lm_engine.check106(img, f);

            f.aligned = img; // 用于后续特征值提取
        }

        f.is_real = spoof_res.is_real;

        auto t2 = std::chrono::high_resolution_clock::now();

        LOGI("【detectFace分析】detect_face=%.2fdms ,check106=%.2fms,score:%.2f\n", std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count(), f.score);
    }
    return final_faces;
}

void FaceEngine::generateProposals(const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob,
    const ncnn::Mat& kps_blob, int stride, float threshold,
    int img_w, int img_h, int pad_w, int pad_h, float scale,
    std::vector<FaceInfo>& faces)
{
    int feat_w = det_input_size / stride;
    int feat_h = det_input_size / stride;
    int num_anchors = 2;
    int total = feat_w * feat_h * num_anchors;

    const float* score_ptr = (const float*)score_blob.data;
    const float* bbox_ptr = (const float*)bbox_blob.data;
    const float* kps_ptr = (const float*)kps_blob.data;

    for (int i = 0; i < total; ++i) {
        // ★ 直接单通道读取，你的模型 w=1
        float score = score_ptr[i];

        if (score < threshold)
            continue;

        // 算网格中心点
        int w_idx = (i / num_anchors) % feat_w;
        int h_idx = (i / num_anchors) / feat_w;
        float cx = (w_idx + 0.0f) * stride;
        float cy = (h_idx + 0.0f) * stride;

        // ★ 核心修复：第 i 个 anchor 的 4 个值连续读取
        float dx = bbox_ptr[i * bbox_blob.w + 0];
        float dy = bbox_ptr[i * bbox_blob.w + 1];
        float dw = bbox_ptr[i * bbox_blob.w + 2];
        float dh = bbox_ptr[i * bbox_blob.w + 3];

        float x1 = cx - dx * stride;
        float y1 = cy - dy * stride;
        float x2 = cx + dw * stride;
        float y2 = cy + dh * stride;

        if (x2 <= x1 || y2 <= y1)
            continue;

        // 减 pad 并还原到原图
        float rx1 = (x1 - pad_w) / scale;
        float ry1 = (y1 - pad_h) / scale;
        float rx2 = (x2 - pad_w) / scale;
        float ry2 = (y2 - pad_h) / scale;

        rx1 = std::max(0.0f, std::min((float)img_w, rx1));
        ry1 = std::max(0.0f, std::min((float)img_h, ry1));
        rx2 = std::max(0.0f, std::min((float)img_w, rx2));
        ry2 = std::max(0.0f, std::min((float)img_h, ry2));

        if (rx2 - rx1 < 5 || ry2 - ry1 < 5)
            continue;

        FaceInfo info;
        info.score = score;
        info.rect = cv::Rect_<float>(rx1, ry1, rx2 - rx1, ry2 - ry1);

        // ★ 修复 kps 读取
        for (int j = 0; j < 5; ++j) {
            float kx = kps_ptr[i * kps_blob.w + (j * 2 + 0)];
            float ky = kps_ptr[i * kps_blob.w + (j * 2 + 1)];

            float pkx = cx + kx * stride;
            float pky = cy + ky * stride;

            info.landmarks[j].x = (pkx - pad_w) / scale;
            info.landmarks[j].y = (pky - pad_h) / scale;
        }
        faces.push_back(info);
    }
}

void FaceEngine::extract_feature(FaceInfo& face)
{
    std::vector<float> feature;
    if (face.aligned.empty())
        return;

    cv::Mat src = face.aligned;
    face.alignFace(src, STANDARD_LANDMARKS, rec_input_size);

    ncnn::Mat in = ncnn::Mat::from_pixels(face.aligned.data, ncnn::Mat::PIXEL_BGR, rec_input_size, rec_input_size);
    const float mean[3] = { 127.5f, 127.5f, 127.5f };
    const float norm[3] = { 1.f / 128.f, 1.f / 128.f, 1.f / 128.f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = rec_net.create_extractor();
    ex.input(rec_input_name.c_str(), in);
    ncnn::Mat out;
    ex.extract(rec_output_name.c_str(), out);

    feature.resize(out.w);
    float sum_sq = 0.f;
    for (int i = 0; i < out.w; ++i) {
        feature[i] = out[i];
        sum_sq += feature[i] * feature[i];
    }
    float norm_val = std::sqrt(sum_sq + 1e-6f);
    for (float& v : feature)
        v /= norm_val;

    face.feature = feature;
}

float FaceEngine::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty())
        return 0.f;
    float dot = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        dot += a[i] * b[i];
    return dot;
}
