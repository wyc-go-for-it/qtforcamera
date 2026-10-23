#ifndef WORKTHREAD_H
#define WORKTHREAD_H

#include "productdatabase.h"
#include "producttextcleaner.h"
#include "qthread.h"

#include <QCamera>
#include <QMutex>
#include <QVideoFrame>

#include <opencv2/core/mat.hpp>

class OcrLite;
class BboxU2netp;
class ISNetSegmenter;
class LiteSaliencyNet;
class Yolo11Ncnn;
class HandDetector;

struct RecogResult {
    uint64_t tempId;
    uint64_t id;
    std::string name;
    std::string barcode;
    float similarity;
    size_t recordIndex;

    bool show = false;
};
Q_DECLARE_METATYPE(RecogResult);

struct DetectObject {
    QRectF rect; // 目标框 [x, y, width, height]
    QString name; // 类别
    float prob; // 置信度
};
Q_DECLARE_METATYPE(DetectObject);

class WorkThread : public QObject {
    Q_OBJECT

public:
    explicit WorkThread(QObject* parent = nullptr);
    ~WorkThread();

    void init();
    void studying(const ProductRecord& record);

public slots:
    void onVideoFrameChanged(const QVideoFrame& frame);
    void onRecognition(bool recog);
    void onUpdateVertex(QVector<QPointF> points, QCamera::Position pos, qint32 rotationAngle);

signals:
    void cropROIed(const QImage& frame, const QImage& diff, const QImage& undiff, const QImage& binary_diff);
    void error(const QString& error);
    void recogFinised(const QList<RecogResult>& data);

    void detected(const QVector<DetectObject>& objects);

private:
    void studied(const cv::Mat& cur_fg);
    void recognition(const cv::Mat& cur_fg);
    QString modelDir();
    QString ncnnModelDir();
    void initDir();
    cv::Mat cropVideoFrame(QVideoFrame& frame);
    void loadBG();
    void initOcrModel();

    QString ensureModelFile(const QString& qrcPath, const QString& fileName);

    std::string detectText(const cv::Mat& cur_fg);

    void initYoloModel();

    Q_INVOKABLE void handleFrame();

    void initU2netpModel();

    void initISNetModel();

    void initLiteSaliencyNetModel();

    void initYoloHandModel();

private:
    QThread m_thread;
    QAtomicInt isRecognition, isStudied;
    cv::Mat m_bg;
    QList<QVideoFrame> m_frame_lst;
    QMutex m_lock;
    bool m_hanlde_frame = false;

    ProductDatabase m_productDatabase;
    ProductRecord m_studiedRecord;

    std::unique_ptr<OcrLite> m_ocr;

    ProductTextCleaner m_textCleaner;

    QVector<QPointF> m_cropPoints;
    cv::Rect m_cropRect;

    std::unique_ptr<Yolo11Ncnn> m_yoloDetector;

    QCamera::Position m_mirrored = QCamera::UnspecifiedPosition;
    int m_rotationAngle = 0;

    std::unique_ptr<BboxU2netp> m_u2netp;

    std::unique_ptr<ISNetSegmenter> m_ISNetSegmenter;

    std::unique_ptr<LiteSaliencyNet> m_LiteSaliencyNet;

    std::unique_ptr<HandDetector> m_yoloDetectorHand;
};

#endif // WORKTHREAD_H
