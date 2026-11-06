#ifndef FACEINFO_H
#define FACEINFO_H
#include "opencv2/imgproc.hpp"
#include <bitset>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

struct FaceInfo {
    cv::Rect_<float> rect;
    float score;
    cv::Point2f landmarks[5];
    std::vector<float> feature;
    cv::Mat aligned;
    std::vector<cv::Point2f> lm106;
    cv::Mat M;

private:
    enum class StatusBit : std::size_t {
        MouthOpen = 0,
        LeftEyeClosed = 1,
        RightEyeClosed = 2,
        HeadTilted = 3, // 歪头
        HeadUp = 4, // 仰头
        HeadDown = 5, // 低头
        HeadTurnLeft = 6, // 左转
        HeadTurnRight = 7, // 右转
    };

    static constexpr std::size_t kStatusBits = 8;

    // 默认全 0（全部 false）
    std::bitset<kStatusBits> status {};

    bool test(StatusBit b) const noexcept
    {
        return status.test(static_cast<std::size_t>(b));
    }

    void set(StatusBit b, bool v = true) noexcept
    {
        status.set(static_cast<std::size_t>(b), v);
    }

public:
    // ---------- mouth_open ----------
    bool isMouthOpen() const noexcept
    {
        return test(StatusBit::MouthOpen);
    }
    void setMouthOpen(bool v = true) noexcept
    {
        set(StatusBit::MouthOpen, v);
    }

    // ---------- eye_closed ----------
    bool isLeftEyeClosed() const noexcept
    {
        return test(StatusBit::LeftEyeClosed);
    }
    bool isRightEyeClosed() const noexcept
    {
        return test(StatusBit::RightEyeClosed);
    }
    void setLeftEyeClosed(bool v = true) noexcept
    {
        set(StatusBit::LeftEyeClosed, v);
    }
    void setRightEyeClosed(bool v = true) noexcept
    {
        set(StatusBit::RightEyeClosed, v);
    }
    bool anyEyeClosed() const noexcept
    {
        return isLeftEyeClosed() || isRightEyeClosed();
    }
    bool allEyeClosed() const noexcept
    {
        return isLeftEyeClosed() && isRightEyeClosed();
    }

    // ---------- head_status ----------
    bool isHeadTilted() const noexcept
    {
        return test(StatusBit::HeadTilted);
    }
    bool isHeadUp() const noexcept
    {
        return test(StatusBit::HeadUp);
    }
    bool isHeadDown() const noexcept
    {
        return test(StatusBit::HeadDown);
    }
    bool isHeadTurnLeft() const noexcept
    {
        return test(StatusBit::HeadTurnLeft);
    }
    bool isHeadTurnRight() const noexcept
    {
        return test(StatusBit::HeadTurnRight);
    }

    void setHeadTilted(bool v = true) noexcept
    {
        set(StatusBit::HeadTilted, v);
    }
    void setHeadUp(bool v = true) noexcept
    {
        set(StatusBit::HeadUp, v);
    }
    void setHeadDown(bool v = true) noexcept
    {
        set(StatusBit::HeadDown, v);
    }
    void setHeadTurnLeft(bool v = true) noexcept
    {
        set(StatusBit::HeadTurnLeft, v);
    }
    void setHeadTurnRight(bool v = true) noexcept
    {
        set(StatusBit::HeadTurnRight, v);
    }

    void alignFace(const cv::Mat& img, const std::array<std::array<float, 2>, 5>& standard_pts, size_t target_size)
    {
        // 构造相似变换的最小二乘线性方程组 AX = B
        // 未知数 X = [a, b, tx, ty]，其中 a = s*cos(theta), b = s*sin(theta)
        cv::Mat A(10, 4, CV_32F);
        cv::Mat B(10, 1, CV_32F);

        const auto& landmarks = this->landmarks;

        for (int i = 0; i < 5; ++i) {
            float x = landmarks[i].x;
            float y = landmarks[i].y;
            float xp = standard_pts[i][0];
            float yp = standard_pts[i][1];

            // 方程 1: x*a - y*b + tx = xp
            A.at<float>(2 * i, 0) = x;
            A.at<float>(2 * i, 1) = -y;
            A.at<float>(2 * i, 2) = 1.0f;
            A.at<float>(2 * i, 3) = 0.0f;
            B.at<float>(2 * i, 0) = xp;

            // 方程 2: y*a + x*b + ty = yp
            A.at<float>(2 * i + 1, 0) = y;
            A.at<float>(2 * i + 1, 1) = x;
            A.at<float>(2 * i + 1, 2) = 0.0f;
            A.at<float>(2 * i + 1, 3) = 1.0f;
            B.at<float>(2 * i + 1, 0) = yp;
        }

        // 使用奇异值分解（SVD）求解最小二乘解 AX = B
        cv::Mat X;
        cv::solve(A, B, X, cv::DECOMP_SVD);

        float a = X.at<float>(0, 0);
        float b = X.at<float>(1, 0);
        float tx = X.at<float>(2, 0);
        float ty = X.at<float>(3, 0);

        // 组装成 2x3 仿射变换矩阵
        this->M = (cv::Mat_<float>(2, 3) << a, -b, tx, b, a, ty);

        // 执行仿射变换
        cv::warpAffine(img, this->aligned, this->M, cv::Size(target_size, target_size), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
};

#endif // FACEINFO_H
