#include "frameConverter.h"
#include <QDebug>
#include <QElapsedTimer>

namespace {

// ---------- YUV -> RGB 查表（BT.601） ----------

struct Yuv2RgbTable {
    int y[256], rv[256], gu[256], gv[256], bu[256];

    void initLimited()
    {
        for (int i = 0; i < 256; ++i) {
            y[i] = 298 * (i - 16);
            rv[i] = 409 * (i - 128);
            gu[i] = -100 * (i - 128);
            gv[i] = -208 * (i - 128);
            bu[i] = 516 * (i - 128);
        }
    }
    void initFull()
    {
        for (int i = 0; i < 256; ++i) {
            y[i] = 256 * i;
            rv[i] = 359 * (i - 128);
            gu[i] = -88 * (i - 128);
            gv[i] = -183 * (i - 128);
            bu[i] = 454 * (i - 128);
        }
    }
};

const Yuv2RgbTable& limitedTable()
{
    static const Yuv2RgbTable t = [] { Yuv2RgbTable x; x.initLimited(); return x; }();
    return t;
}
const Yuv2RgbTable& fullTable()
{
    static const Yuv2RgbTable t = [] { Yuv2RgbTable x; x.initFull(); return x; }();
    return t;
}

inline int clip8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

inline QRgb yuv2rgb(int Y, int U, int V, const Yuv2RgbTable& t)
{
    const int yv = t.y[Y];
    return qRgb(clip8((yv + t.rv[V] + 128) >> 8),
        clip8((yv + t.gv[V] + t.gu[U] + 128) >> 8),
        clip8((yv + t.bu[U] + 128) >> 8));
}

// ---------- 兜底：任何无法可靠拆平面的格式，走 image() ----------

FrameData fromImage(const QImage& img, qint64 startTime)
{
    FrameData d;
    if (img.isNull())
        return d;

    QImage rgb = img.format() != QImage::Format_RGBA8888 ? img.convertToFormat(QImage::Format_RGBA8888) : img;

    d.width = rgb.width();
    d.height = rgb.height();
    d.format = QVideoFrame::Format_RGB32;
    d.startTime = startTime;
    d.planes.append(QByteArray(
        reinterpret_cast<const char*>(rgb.constBits()),
        rgb.sizeInBytes()));
    d.strides.append(rgb.bytesPerLine());
    return d;
}

} // namespace

// ================= convert =================

FrameData FrameConverter::convert(const QVideoFrame& frame)
{
    FrameData d;
    if (!frame.isValid())
        return d;

    const auto fmt = frame.pixelFormat();
    const qint64 st = frame.startTime();

    if (fmt == QVideoFrame::Format_Jpeg) {
        return fromImage(frame.image(), st);
    }

    QVideoFrame in(frame);
    if (!in.map(QAbstractVideoBuffer::ReadOnly)) {
        return fromImage(frame.image(), st);
    }

    d.width = in.width();
    d.height = in.height();
    d.format = fmt;
    d.startTime = st;

    const uchar* base = in.bits();
    const int stride = in.bytesPerLine();
    const int h = d.height;
    const int w = d.width;

    switch (fmt) {

        // -------- NV12 / NV21：Y 平面 + VU/UV 交错平面，紧接其后 --------
    case QVideoFrame::Format_NV12:
    case QVideoFrame::Format_NV21: {
        d.planes.append(QByteArray(reinterpret_cast<const char*>(base), stride * h));
        d.strides.append(stride);

        const int uvH = (h + 1) / 2;
        const uchar* uv = base + stride * h;
        d.planes.append(QByteArray(reinterpret_cast<const char*>(uv), stride * uvH));
        d.strides.append(stride);
        break;
    }

        // -------- packed YUV：单平面 --------
    case QVideoFrame::Format_YUYV:
    case QVideoFrame::Format_UYVY: {
        d.planes.append(QByteArray(reinterpret_cast<const char*>(base), stride * h));
        d.strides.append(stride);
        break;
    }

        // -------- 单平面 RGB / 灰度 --------
    case QVideoFrame::Format_RGB32:
    case QVideoFrame::Format_ARGB32:
    case QVideoFrame::Format_ARGB32_Premultiplied:
    case QVideoFrame::Format_RGB24:
    case QVideoFrame::Format_RGB565:
    case QVideoFrame::Format_Y8:
    case QVideoFrame::Format_Y16: {
        d.planes.append(QByteArray(reinterpret_cast<const char*>(base), stride * h));
        d.strides.append(stride);
        break;
    }

        // -------- YUV420P/YV12/YUV422P/YUV444：Qt5.15 无多平面 API，不可靠 --------
    default: {
        in.unmap();
        return fromImage(frame.image(), st);
    }
    }

    in.unmap();
    return d;
}

// ================= toImage =================

QImage FrameConverter::toImage(const FrameData& d, bool fullRange)
{
    if (!d.isValid())
        return QImage();

    switch (d.format) {
    case QVideoFrame::Format_RGB32:
    case QVideoFrame::Format_ARGB32:
    case QVideoFrame::Format_ARGB32_Premultiplied:
        return QImage(reinterpret_cast<const uchar*>(d.planes[0].constData()),
            d.width, d.height, d.strides[0],
            QImage::Format_RGB32);

    case QVideoFrame::Format_Y8:
        return QImage(reinterpret_cast<const uchar*>(d.planes[0].constData()),
            d.width, d.height, d.strides[0],
            QImage::Format_Grayscale8);

    case QVideoFrame::Format_RGB24: {
        QImage view(reinterpret_cast<const uchar*>(d.planes[0].constData()),
            d.width, d.height, d.strides[0],
            QImage::Format_RGB888);
        return view.convertToFormat(QImage::Format_RGB32);
    }

    default:
        break;
    }

    // ---------- YUV 查表 ----------
    const bool yuv = (d.format == QVideoFrame::Format_NV21
        || d.format == QVideoFrame::Format_NV12
        || d.format == QVideoFrame::Format_YUYV
        || d.format == QVideoFrame::Format_UYVY);
    if (!yuv) {
        qWarning() << "FrameConverter::toImage: unsupported format" << d.format;
        return QImage();
    }

    QImage img(d.width, d.height, QImage::Format_RGB32);
    if (img.isNull())
        return img;

    const int w = d.width, h = d.height;
    const Yuv2RgbTable& t = fullRange ? fullTable() : limitedTable();
    const uchar* yPlane = reinterpret_cast<const uchar*>(d.planes[0].constData());
    const int yStride = d.strides[0];

    switch (d.format) {

    case QVideoFrame::Format_NV21:
    case QVideoFrame::Format_NV12: {
        const uchar* uvPlane = reinterpret_cast<const uchar*>(d.planes[1].constData());
        const int uvStride = d.strides[1];
        const bool vFirst = (d.format == QVideoFrame::Format_NV21);

        for (int y = 0; y < h; ++y) {
            const uchar* yRow = yPlane + y * yStride;
            const uchar* uvRow = uvPlane + (y >> 1) * uvStride;
            QRgb* dst = reinterpret_cast<QRgb*>(img.scanLine(y));

            for (int x = 0; x < w; ++x) {
                const uchar* uv = uvRow + ((x >> 1) << 1);
                const int U = vFirst ? uv[1] : uv[0];
                const int V = vFirst ? uv[0] : uv[1];
                dst[x] = yuv2rgb(yRow[x], U, V, t);
            }
        }
        break;
    }

    case QVideoFrame::Format_YUYV:
    case QVideoFrame::Format_UYVY: {
        const bool yuyv = (d.format == QVideoFrame::Format_YUYV);

        for (int y = 0; y < h; ++y) {
            const uchar* row = yPlane + y * yStride;
            QRgb* dst = reinterpret_cast<QRgb*>(img.scanLine(y));

            for (int x = 0; x < w; x += 2) {
                int Y0, Y1, U, V;
                if (yuyv) {
                    Y0 = row[0];
                    U = row[1];
                    Y1 = row[2];
                    V = row[3];
                } else {
                    U = row[0];
                    Y0 = row[1];
                    V = row[2];
                    Y1 = row[3];
                }

                dst[x] = yuv2rgb(Y0, U, V, t);
                if (x + 1 < w)
                    dst[x + 1] = yuv2rgb(Y1, U, V, t);
                row += 4;
            }
        }
        break;
    }

    default:
        break;
    }

    return img;
}

QImage FrameConverter::toImageCopy(const FrameData& d, bool fullRange)
{
    QImage img = toImage(d, fullRange);
    return img.isNull() ? img : img.copy();
}

QImage FrameConverter::toGrayscale(const FrameData& d)
{
    if (!d.isValid())
        return QImage();

    switch (d.format) {
    case QVideoFrame::Format_NV21:
    case QVideoFrame::Format_NV12:
    case QVideoFrame::Format_Y8:
        return QImage(reinterpret_cast<const uchar*>(d.planes[0].constData()),
            d.width, d.height, d.strides[0],
            QImage::Format_Grayscale8);
    default:
        return QImage();
    }
}
