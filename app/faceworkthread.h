#ifndef FACEWORKTHREAD_H
#define FACEWORKTHREAD_H

#include "workthread.h"

#include <QCamera>
#include <QFileInfo>
#include <QMutex>
#include <QObject>
#include <QThread>
#include <QVideoFrame>

#include <opencv2/core/mat.hpp>

class FaceEngine;

class FaceWorkThread : public QObject {
    Q_OBJECT

public:
    explicit FaceWorkThread(QObject* parent = nullptr);
    ~FaceWorkThread();

    void init();

public slots:
    void onVideoFrameChanged(const QVideoFrame& frame);
    void onRegisteredFace(bool recog, const QString& name);
    void onQueriedFace(bool recog, const QString& name);
    void onUpdateVertex(QVector<QPointF> points, QCamera::Position pos, qint32 rotationAngle);
    void onExtractFeature();

signals:
    void captured(const QImage& frame, const QImage& alinged);
    void error(const QString& error);
    void detected(const QVector<DetectObject>& objects);

    void info(const QString& error);

private:
    void registerFace(const cv::Mat& cur_fg, const QString& name = {});
    void queryFace(const cv::Mat& cur_fg);
    void initDir();
    cv::Mat cropVideoFrame(QVideoFrame& frame);

    Q_INVOKABLE void handleFrame();
    Q_INVOKABLE void extractFeature(const QFileInfo& fileInfo);

    void initFaceModel();

    QString faceDataDir();

private:
    QThread m_thread;
    QAtomicInt isRegistered, isQueried;

    cv::Mat m_bg;
    QList<QVideoFrame> m_frame_lst;
    QMutex m_lock;
    bool m_hanlde_frame = false;

    QVector<QPointF> m_cropPoints;
    cv::Rect m_cropRect;

    QCamera::Position m_mirrored = QCamera::UnspecifiedPosition;
    int m_rotationAngle = 0;

    std::unique_ptr<FaceEngine> m_faceDetector;

    std::unique_ptr<ProductDatabase> m_FaceDatabase;

    QString m_name;
};

#endif // FACEWORKTHREAD_H
