#ifndef CUSTOMGLSURFACE_H
#define CUSTOMGLSURFACE_H

#include <QAbstractVideoSurface>
#include <QVideoFrame>

class CustomGLSurface : public QAbstractVideoSurface {
    Q_OBJECT
public:
    explicit CustomGLSurface(QObject* parent = nullptr)
        : QAbstractVideoSurface(parent)
    {
    }

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType = QAbstractVideoBuffer::NoHandle) const override;
    bool present(const QVideoFrame& frame) override;

signals:
    void frameAvailable(const QVideoFrame& frame);
};

#endif // CUSTOMGLSURFACE_H
