#include "roiviewfinder.h"

ROIViewfinder::~ROIViewfinder()
{
    savePoints();
}

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

        emit polygonChanged(calOriginalPoints());
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

            emit polygonChanged(calOriginalPoints());
        }
    }
    QCameraViewfinder::mouseMoveEvent(event);
}

QVector<QPointF> ROIViewfinder::calOriginalPoints()
{
    QVector<QPointF> imgPts;
    const QSize& imageSize = m_actualImageSize;
    if (m_points.empty() || imageSize.isEmpty())
        return imgPts;

    QSize viewfinderSize = this->size();

    QSize scaledSize = imageSize.scaled(viewfinderSize, Qt::KeepAspectRatio);
    double targetX = (viewfinderSize.width() - scaledSize.width()) / 2.0;
    double targetY = (viewfinderSize.height() - scaledSize.height()) / 2.0;

    double scaleX = static_cast<double>(imageSize.width()) / scaledSize.width();
    double scaleY = static_cast<double>(imageSize.height()) / scaledSize.height();

    for (const auto& pt : qAsConst(m_points)) {

        double imgX = (pt.x() - targetX) * scaleX;
        double imgY = (pt.y() - targetY) * scaleY;

        imgX = std::clamp(imgX, 0.0, static_cast<double>(imageSize.width() - 1));
        imgY = std::clamp(imgY, 0.0, static_cast<double>(imageSize.height() - 1));

        imgPts.push_back(QPointF(imgX, imgY));
    }

    return imgPts;
}

void ROIViewfinder::savePoints()
{
    QSettings settings("./config/hz_config", QSettings::Format::IniFormat);

    settings.beginWriteArray("roi_points");
    for (int i = 0; i < m_points.size(); ++i) {
        settings.setArrayIndex(i);
        settings.setValue("point", m_points[i]);
    }
    settings.endArray();
}

QVector<QPointF> ROIViewfinder::loadPoints()
{
    QSettings settings("./config/hz_config", QSettings::Format::IniFormat);
    QVector<QPointF> points;

    int size = settings.beginReadArray("roi_points");
    for (int i = 0; i < size; ++i) {
        settings.setArrayIndex(i);
        QPointF pt = settings.value("point").toPointF();
        points.append(pt);
    }
    settings.endArray();

    return points;
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

    painter.setPen(QPen(Qt::red, 1, Qt::DotLine));
    painter.setBrush(QColor(255, 0, 0, 25));

    painter.drawRect(poly.boundingRect());

    painter.setPen(QPen(Qt::green, 1, Qt::SolidLine));
    painter.setBrush(Qt::NoBrush);
    painter.drawPolygon(poly);

    for (int i = 0; i < 4; ++i) {
        painter.setPen(QPen(Qt::white, 2));
        painter.setBrush(i == m_draggedIndex ? Qt::yellow : Qt::green);
        painter.drawEllipse(m_points[i], 8, 8);
    }
}

void ROIViewfinder::initDefaultPoints()
{

    m_points = loadPoints();

    double w = width() > 0 ? width() : 640;
    double h = height() > 0 ? height() : 480;

    if (m_points.isEmpty()) {
        m_points.resize(4);

        m_points[0] = QPointF(w * 0.2, h * 0.2); // 左上
        m_points[1] = QPointF(w * 0.8, h * 0.2); // 右上
        m_points[2] = QPointF(w * 0.85, h * 0.8); // 右下 (稍宽，模拟近大远小)
        m_points[3] = QPointF(w * 0.15, h * 0.8); // 左下
    }
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
