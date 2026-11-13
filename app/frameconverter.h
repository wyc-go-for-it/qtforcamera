#pragma once

#include <QByteArray>
#include <QImage>
#include <QVector>
#include <QVideoFrame>

struct FrameData {
    int width = 0;
    int height = 0;
    QVideoFrame::PixelFormat format = QVideoFrame::Format_Invalid;
    qint64 startTime = 0;

    QVector<QByteArray> planes;
    QVector<int> strides;

    bool isValid() const
    {
        return width > 0 && height > 0
            && !planes.isEmpty() && !planes[0].isEmpty()
            && strides.size() == planes.size();
    }
};

class FrameConverter {
public:
    // QVideoFrame -> FrameData。map 窗口只包 memcpy
    static FrameData convert(const QVideoFrame& frame);

    // FrameData -> QImage
    // RGB32/ARGB32/Y8：零拷贝视图，调用方须保证 d 存活
    // RGB24：转 RGB32（深拷贝）
    // NV12/NV21/YUYV/UYVY：查表转 RGB32
    // fullRange 为 true 时用 BT.601 full 系数，默认 limited
    static QImage toImage(const FrameData& d, bool fullRange = false);

    static QImage toImageCopy(const FrameData& d, bool fullRange = false);

    // YUV 格式取 Y plane 作为灰度视图（零拷贝，最快路径）
    static QImage toGrayscale(const FrameData& d);
};
