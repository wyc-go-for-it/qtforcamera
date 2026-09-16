#include <QCameraViewfinder>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>

class ROIViewfinder : public QCameraViewfinder {
    Q_OBJECT
public:
    explicit ROIViewfinder(QWidget* parent = nullptr)
        : QCameraViewfinder(parent)
    {
        initDefaultPoints();
    }

    QVector<QPointF> getPolygonPoints() const { return m_points; }

signals:
    void polygonChanged(const QVector<QPointF>& points);

protected:
    void resizeEvent(QResizeEvent* event) override;

    void mousePressEvent(QMouseEvent* event) override;

    void mouseMoveEvent(QMouseEvent* event) override;

    void mouseReleaseEvent(QMouseEvent* event) override;

    void paintEvent(QPaintEvent* event) override;

private:
    void initDefaultPoints();

    int getHitPointIndex(const QPointF& mousePos);

private:
    QVector<QPointF> m_points; // 0:左上, 1:右上, 2:右下, 3:左下
    int m_draggedIndex = -1;

    bool m_isDraggingVertex = false; // 是否在拖拽单个顶点
    bool m_isDraggingPolygon = false; // 是否在拖拽整个图形
    QPointF m_lastMousePos; // 拖拽整体时的上一帧鼠标位置
};
