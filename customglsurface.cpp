#include "customglsurface.h"

QList<QVideoFrame::PixelFormat> CustomGLSurface::supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const
{
    Q_UNUSED(handleType);
    return QList<QVideoFrame::PixelFormat>()
        << QVideoFrame::Format_NV21
        << QVideoFrame::Format_NV12
        << QVideoFrame::Format_YUV420P
        << QVideoFrame::Format_YUYV;
}

bool CustomGLSurface::present(const QVideoFrame& frame)
{
    if (!frame.isValid())
        return false;
    emit frameAvailable(frame);
    return true;
}
