#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <QImage>
#include <opencv2/opencv.hpp>

class ImageUtils {
public:
    static QImage matToQImage(const cv::Mat& mat, bool clone = true);
    static cv::Mat qImageToMat(const QImage& image, bool clone = true);
};

#endif // IMAGEUTILS_H
