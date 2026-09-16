#include "ImageUtils.h"

QImage ImageUtils::matToQImage(const cv::Mat& mat, bool clone)
{
    if (mat.empty()) {
        return QImage();
    }

    QImage image;

    switch (mat.type()) {
    // 1. 8位 3通道 BGR 图
    case CV_8UC3: {
        image = QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_BGR888);
        break;
    }
    // 2. 8位 4通道 BGRA 图
    case CV_8UC4: {
        image = QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
        break;
    }
    // 3. 8位 单通道灰度图
    case CV_8UC1: {
        image = QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
        break;
    }
    default:
        // 其他不兼容格式可在此扩展或转换
        return QImage();
    }

    // 若 clone 为 true 则执行深拷贝，确保 cv::Mat 释放后 QImage 依然有效
    return clone ? image.copy() : image;
}

cv::Mat ImageUtils::qImageToMat(const QImage& image, bool clone)
{
    if (image.isNull()) {
        return cv::Mat();
    }

    cv::Mat mat;

    switch (image.format()) {
    // 1. 8位 4通道 RGBA/ARGB
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC4,
            const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        break;
    }
    // 2. 8位 3通道 RGB
    case QImage::Format_RGB888: {
        // 注意：OpenCV 原生要求 BGR，因此需要转一下通道
        QImage swapped = image.rgbSwapped();
        return cv::Mat(swapped.height(), swapped.width(), CV_8UC3,
            const_cast<uchar*>(swapped.bits()), static_cast<size_t>(swapped.bytesPerLine()))
            .clone();
    }
    // 3. 8位 3通道 BGR (Qt5.14+ / Qt6 支持)
    case QImage::Format_BGR888: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC3,
            const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        break;
    }
    // 4. 8位 灰度图
    case QImage::Format_Grayscale8:
    case QImage::Format_Indexed8: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC1,
            const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        break;
    }
    default: {
        // 遇到其他复杂格式，先统转为 ARGB32 再处理
        QImage converted = image.convertToFormat(QImage::Format_ARGB32);
        return qImageToMat(converted, true);
    }
    }

    return clone ? mat.clone() : mat;
}
