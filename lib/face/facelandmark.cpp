#include "FaceLandmark.h"
#include "faceengine.h"

#include <cmath>

#ifdef ENABLE_LOG
#ifdef __ANDROID__
#include "ncnn/cpu.h"
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

bool FaceLandmark::load_model(const std::string& param, const std::string& bin)
{
#if defined(__ANDROID__)
    net.opt.num_threads = ncnn::get_big_cpu_count();

    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
#else
    net.opt.num_threads = 2;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
#endif

    net.opt.blob_allocator = 0;
    net.opt.workspace_allocator = 0;
    net.opt.openmp_blocktime = 0;

    net.opt.use_packing_layout = true;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;

    net.opt.lightmode = true;

    if (net.load_param(param.c_str()) != 0) {
        LOGE("[FaceLandmark] load_param failed: %s\n", param.c_str());
        return false;
    }
    if (net.load_model(bin.c_str()) != 0) {
        LOGE("[FaceLandmark] load_model failed: %s\n", bin.c_str());
        return false;
    }

    return true;
}

void FaceLandmark::check106(const cv::Mat& src, FaceInfo& face)
{
    face.alignFace(src, FaceLandmark::STANDARD_106_LANDMARKS, 192);

    cv::Mat aligned_192 = face.aligned;

    if (aligned_192.empty()) {
        return;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels(
        aligned_192.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        input_size,
        input_size);

    // in.substract_mean_normalize(mean, norm);//模型中已包含归一化算子

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) {
        return;
    }

    std::vector<cv::Point2f> result;

    const float* ptr = (const float*)out.data;
    for (int i = 0; i < 106; ++i) {
        float raw_x = ptr[i * 2];
        float raw_y = ptr[i * 2 + 1];

        // 模型输出是 [-1, 1] 范围的归一化坐标，将其映射回 192 像素空间
        float x = (raw_x + 1.0f) * 0.5f * input_size;
        float y = (raw_y + 1.0f) * 0.5f * input_size;

        result.emplace_back(x, y);
    }

    auto& lm106_original = face.lm106;

    auto aligned = face.aligned;

    cv::Mat M;
    cv::invertAffineTransform(face.M, M);

    for (const auto& pt : result) {

        float x_aligned = pt.x * ((float)aligned.cols / aligned_192.cols);
        float y_aligned = pt.y * ((float)aligned.rows / aligned_192.rows);

        std::vector<cv::Point2f> src_pts = { cv::Point2f(x_aligned, y_aligned) };
        std::vector<cv::Point2f> dst_pts;
        cv::transform(src_pts, dst_pts, M);

        lm106_original.push_back(dst_pts[0]);
    }

    check_eye_closed(face);
    check_mouth_open(face);
    check_head_pose(face);
}

// 计算距离辅助
static float calc_dist(const cv::Point2f& p1, const cv::Point2f& p2)
{
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

void FaceLandmark::check_eye_closed(FaceInfo& face)
{
    auto lm = face.lm106;

    if (lm.size() != 106)
        return;

    // 右眼：横 35~39，竖 40~33
    float right_w = calc_dist(lm[35], lm[39]);
    float right_h = calc_dist(lm[40], lm[33]);

    // 左眼：横 99~93，竖 94~97
    float left_w = calc_dist(lm[89], lm[93]);
    float left_h = calc_dist(lm[94], lm[87]);

    if (left_w < 1e-6f || right_w < 1e-6f)
        return;

    float ear_left = left_h / left_w;
    float ear_right = right_h / right_w;

    // LOGI("左眼 宽:%.2f 高:%.2f EAR:%.3f | 右眼 宽:%.2f 高:%.2f EAR:%.3f\n", left_w, left_h, ear_left, right_w, right_h, ear_right);

    // 经验阈值：正常睁眼 0.25~0.35，闭眼 < 0.15
    face.setLeftEyeClosed(ear_left < 0.18f);
    face.setRightEyeClosed(ear_right < 0.18f);
}

void FaceLandmark::check_mouth_open(FaceInfo& face)
{
    auto lm = face.lm106;

    if (lm.size() != 106)
        return;

    // 嘴巴：左嘴角 52，右嘴角 61
    float mouth_w = calc_dist(lm[52], lm[61]);
    // 嘴巴：上唇中心 71，下唇中心 53
    float mouth_h = calc_dist(lm[71], lm[53]);

    if (mouth_w < 1e-6f)
        return;

    float mar = mouth_h / mouth_w;

    // LOGI("嘴巴 宽:%.2f 高:%.2f MAR:%.3f\n", mouth_w, mouth_h, mar);

    // 经验阈值：闭嘴 0.05~0.15，张嘴 > 0.4
    face.setMouthOpen(mar > 0.45f);
}

void FaceLandmark::check_head_pose(FaceInfo& face)
{
    auto lm = face.lm106;

    if (lm.size() != 106)
        return;

    cv::Point2f left_eye = lm[35];
    cv::Point2f right_eye = lm[99];
    cv::Point2f nose = lm[86];
    cv::Point2f chin = lm[0];

    // 1. 算两只眼睛的中心点
    cv::Point2f eye_center = (left_eye + right_eye) * 0.5f;

    // 2. 算歪头角度 (Roll)
    float dx = right_eye.x - left_eye.x;
    float dy = right_eye.y - left_eye.y;
    float roll_angle_rad = std::atan2(dy, dx); // 弧度制

    // 3. ★ 核心：在数学上把鼻子和眼睛反向旋转，消除歪头的影响
    // 把坐标平移到以眼睛中心为原点
    float nose_rel_x = nose.x - eye_center.x;
    float nose_rel_y = nose.y - eye_center.y;
    float left_rel_x = left_eye.x - eye_center.x;
    float right_rel_x = right_eye.x - eye_center.x;

    // 旋转 -roll_angle，把脸拉平
    float cos_a = std::cos(-roll_angle_rad);
    float sin_a = std::sin(-roll_angle_rad);

    // 旋转后的鼻子 X 坐标
    float rotated_nose_x = nose_rel_x * cos_a - nose_rel_y * sin_a;
    // 旋转后的眼距
    float rotated_eye_dist = std::abs(right_rel_x * cos_a - (right_eye.y - eye_center.y) * sin_a - (left_rel_x * cos_a - (left_eye.y - eye_center.y) * sin_a));

    // 在摆正的坐标系里，算鼻子的相对水平偏移
    float yaw_offset = rotated_nose_x / (rotated_eye_dist + 1e-6f);

    // 4. 算俯仰比 (Pitch)
    float nose_to_chin = calc_dist(nose, chin);
    float eye_to_chin = calc_dist(eye_center, chin);
    float pitch_ratio = nose_to_chin / (eye_to_chin + 1e-6f);

    // ================= 启动动态校准 =================
    static float base_roll = 0.0f, base_pitch = 0.0f, base_yaw = 0.0f;
    static int calib_frames = 0;
    static bool is_calibrated = false;

    if (!is_calibrated) {
        base_roll += roll_angle_rad * 180.0f / CV_PI;
        base_pitch += pitch_ratio;
        base_yaw += yaw_offset;
        calib_frames++;
        if (calib_frames >= 30) {
            base_roll /= 30.0f;
            base_pitch /= 30.0f;
            base_yaw /= 30.0f;
            is_calibrated = true;
            LOGI("✅ 校准完成: 基准 Roll=%.2f, 基准 Pitch=%.2f, 基准 Yaw=%.2f\n", base_roll, base_pitch, base_yaw);
        }
        return;
    }

    // 计算相对偏移
    float current_roll = roll_angle_rad * 180.0f / CV_PI;
    float delta_roll = current_roll - base_roll;
    float delta_pitch = pitch_ratio - base_pitch;
    float delta_yaw = yaw_offset - base_yaw;

    // 打印当前数值（方便调阈值）
    // LOGI("当前姿态: 歪头=%.2f, 俯仰=%.2f, 转头=%.2f\n", delta_roll, delta_pitch, delta_yaw);

    // ================= 报警判断 =================
    // 1. 歪头 (Roll)：相对基准偏了 20 度以上
    face.setHeadTilted(std::abs(delta_roll) > 20.0f);

    static float threshold_pitch = 0.08f;
    // 2. 俯仰 (Pitch)：相对基准偏移超过 threshold_pitch
    face.setHeadUp(delta_pitch > threshold_pitch);
    face.setHeadDown(delta_pitch < -threshold_pitch);

    static float threshold_yaw = 0.15f;
    // 3. 转头 (Yaw)：相对基准偏移超过 threshold_yaw
    face.setHeadTurnLeft(delta_yaw > threshold_yaw);
    face.setHeadTurnRight(delta_yaw < -threshold_yaw);
}
