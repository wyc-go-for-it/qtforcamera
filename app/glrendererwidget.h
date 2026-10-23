#ifndef QGLRENDERERWIDGET_H
#define QGLRENDERERWIDGET_H

#include "qmutex.h"
#include "workthread.h"
#include <QCursor>
#include <QDebug>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QPainter>
#include <QSettings>
#include <QTransform>
#include <QVideoFrame>

#include <QCamera>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QOpenGLWindow>

#ifdef Q_OS_ANDROID
#include <GLES3/gl3.h>
#endif

enum class ShaderType {
    None,
    NV21,
    YUYV,
    RGB
};

class GLRendererWidget : public QOpenGLWindow, protected QOpenGLFunctions {
    Q_OBJECT
private:
    // OpenGL 资源
    QOpenGLShaderProgram m_programNV21;
    QOpenGLShaderProgram m_programYUYV;
    QOpenGLShaderProgram m_programRGB;

    QOpenGLShaderProgram m_lineProgram;

    ShaderType m_currentShaderType = ShaderType::None;

    GLuint m_textures[2] = { 0, 0 };
    int m_videoWidth = 0; // 视频帧原始宽度
    int m_videoHeight = 0; // 视频帧原始高度

    // 矩阵与旋转
    QMatrix4x4 m_rotationMatrix;
    int m_rotationAngle = 0;

    // 顶点坐标（按比例缩放后）
    GLfloat m_vertexPos[8] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        -1.0f, 1.0f,
        1.0f, 1.0f
    };

    // 鼠标拖拽交互状态
    QVector<QPointF> m_points; // 0:左上, 1:右上, 2:右下, 3:左下
    int m_draggedIndex = -1;

    bool m_isDraggingVertex = false; // 是否在拖拽单个顶点
    bool m_isDraggingPolygon = false; // 是否在拖拽整个图形
    QPointF m_lastMousePos; // 拖拽整体时的上一帧鼠标位置

    QSize m_actualImageSize;

    QCamera::Position m_mirrored = QCamera::UnspecifiedPosition;

    QVector<DetectObject> m_currentObjects;

    bool is_init = false;

public:
    explicit GLRendererWidget(QWindow* parent = nullptr, QOpenGLWindow::UpdateBehavior behavior = NoPartialUpdate);

    ~GLRendererWidget() override;

    void setRotation(int angle);

    void setMirrored(QCamera::Position on)
    {
        m_mirrored = on;
        update();
    }

public slots:
    void updateFrame(const QVideoFrame& frame);
    void updateImageSize(const QSize& size);

    void updateCurrentObjects(const QVector<DetectObject>& objects);

signals:
    void polygonChanged(QVector<QPointF>, QCamera::Position, qint32 rotationAngle);

protected:
    void initializeGL() override;

    void paintGL() override;

    void resizeGL(int w, int h) override;

    void resizeEvent(QResizeEvent* event) override;

    void mousePressEvent(QMouseEvent* event) override;

    void mouseMoveEvent(QMouseEvent* event) override;

    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void initDefaultPoints();

    QVector<QPointF> loadPoints();

    int getHitPointIndex(const QPointF& mousePos);

    QVector<QPointF> calOriginalPoints();

    void savePoints();

    // 计算高宽比居中填充顶点的核心算法
    void updateVertexPos();

    void initShader(QOpenGLShaderProgram& program, const char* vsrc, const char* fsrc);

    std::vector<float> generateBoxVertices();

    void drawYoloBoxes();

    QMutex mlock;
};

#endif // QGLRENDERERWIDGET_H
