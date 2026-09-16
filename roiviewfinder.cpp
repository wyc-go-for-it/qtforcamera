#include "roiviewfinder.h"

void ROIViewfinder::resizeEvent(QResizeEvent* event)
{
    QCameraViewfinder::resizeEvent(event);

    QSize oldSize = event->oldSize();
    QSize newSize = event->size();

    if (oldSize.width() <= 0 || oldSize.height() <= 0 || m_points.isEmpty()) {
        initDefaultPoints();
        return;
    }

    double scaleX = static_cast<double>(newSize.width()) / oldSize.width();
    double scaleY = static_cast<double>(newSize.height()) / oldSize.height();

    for (int i = 0; i < m_points.size(); ++i) {
        m_points[i].setX(m_points[i].x() * scaleX);
        m_points[i].setY(m_points[i].y() * scaleY);
    }

    update();
}

void ROIViewfinder::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_draggedIndex = getHitPointIndex(event->pos());
        if (m_draggedIndex != -1) {
            m_isDraggingVertex = true;
        } else if (QPolygonF(m_points).containsPoint(event->pos(), Qt::OddEvenFill)) {
            m_isDraggingPolygon = true;
            m_lastMousePos = event->pos();
            setCursor(Qt::SizeAllCursor); // 切换为平移手势鼠标样式
        }
    }
    QCameraViewfinder::mousePressEvent(event);
}

void ROIViewfinder::mouseMoveEvent(QMouseEvent* event)
{
    if (m_isDraggingVertex && m_draggedIndex != -1) {

        QPointF pos = event->pos();
        pos.setX(qBound(0.0, pos.x(), (double)width()));
        pos.setY(qBound(0.0, pos.y(), (double)height()));

        m_points[m_draggedIndex] = pos;
        update();

        emit polygonChanged(m_points);
    } else if (m_isDraggingPolygon) {
        QPointF delta = event->pos() - m_lastMousePos;
        bool canMove = true;
        for (const auto& pt : qAsConst(m_points)) {
            QPointF nextPt = pt + delta;
            if (nextPt.x() < 0 || nextPt.x() > width() || nextPt.y() < 0 || nextPt.y() > height()) {
                canMove = false;
                break;
            }
        }
        if (canMove) {
            for (auto& pt : m_points) {
                pt += delta;
            }
            m_lastMousePos = event->pos();
            update();

            emit polygonChanged(m_points);
        }
    }
    QCameraViewfinder::mouseMoveEvent(event);
}

void ROIViewfinder::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDraggingVertex = false;
        m_isDraggingPolygon = false;
        m_draggedIndex = -1;
        unsetCursor();
    }
    QCameraViewfinder::mouseReleaseEvent(event);
}

void ROIViewfinder::paintEvent(QPaintEvent* event)
{
    QCameraViewfinder::paintEvent(event);

    if (m_points.size() != 4)
        return;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    QPolygonF poly(m_points);
    painter.setPen(QPen(Qt::green, 2, Qt::SolidLine));
    painter.setBrush(QColor(0, 255, 0, 25));
    painter.drawPolygon(poly);

    for (int i = 0; i < 4; ++i) {
        painter.setPen(QPen(Qt::white, 2));
        painter.setBrush(i == m_draggedIndex ? Qt::yellow : Qt::green);
        painter.drawEllipse(m_points[i], 8, 8);
    }
}

void ROIViewfinder::initDefaultPoints()
{
    m_points.resize(4);
    double w = width() > 0 ? width() : 640;
    double h = height() > 0 ? height() : 480;

    m_points[0] = QPointF(w * 0.2, h * 0.2); // 左上
    m_points[1] = QPointF(w * 0.8, h * 0.2); // 右上
    m_points[2] = QPointF(w * 0.85, h * 0.8); // 右下 (稍宽，模拟近大远小)
    m_points[3] = QPointF(w * 0.15, h * 0.8); // 左下
}

int ROIViewfinder::getHitPointIndex(const QPointF& mousePos)
{
    for (int i = 0; i < m_points.size(); ++i) {
        double dist = std::hypot(mousePos.x() - m_points[i].x(), mousePos.y() - m_points[i].y());
        if (dist <= 15.0) {
            return i;
        }
    }
    return -1;
}
