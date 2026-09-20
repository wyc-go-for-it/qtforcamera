#include "goodsdelegate.h"

#include <QPainter>

GoodsDelegate::GoodsDelegate(QObject* parent)
    : QStyledItemDelegate { parent }
{
}

void GoodsDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    if (index.isValid()) {
        painter->save();
        painter->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);

        const QRect& base_rect = option.rect;

        m_path.clear();
        m_path.addRoundedRect(base_rect.left(), base_rect.top(), base_rect.width() - 1, base_rect.height() - 1, 5, 5);
        m_path.translate(0.5, 0.5);

        if (option.state.testFlag(QStyle::State_Selected)) {
            QColor status_color { "#2aa99c" };
            painter->setPen(QPen(status_color, 1, Qt::SolidLine, Qt::SquareCap, Qt::RoundJoin));

            painter->drawPath(m_path);

            const auto bottomRight = base_rect.bottomRight();

            auto point1 = QPointF(bottomRight.x(), bottomRight.y() - base_rect.height() * 0.25);
            auto point2 = QPointF(bottomRight.x(), bottomRight.y() - 5);

            auto point3 = QPoint(bottomRight.x() - base_rect.width() * 0.25, bottomRight.y());
            auto point4 = QPoint(bottomRight.x() - 5, bottomRight.y());

            m_path.clear();

            m_path.moveTo(point1);

            m_path.lineTo(point2);

            m_path.cubicTo(bottomRight, bottomRight, point4);

            m_path.lineTo(point3);

            painter->setPen(Qt::NoPen);
            status_color.setAlphaF(0.6);
            painter->setBrush(status_color);

            painter->drawPath(m_path);

        } else {
            painter->setPen(QPen(QColor(200, 200, 200), 1));
            painter->drawPath(m_path);
        }

        painter->setPen(Qt::red);
        painter->drawText(base_rect, Qt::AlignCenter, index.data().toString());

        auto similarity = index.data(Qt::UserRole + 9);
        if (similarity.isValid()) {
            painter->drawText(option.rect.marginsRemoved({ 0, 28, 0, 0 }), Qt::AlignCenter, QString::number(similarity.toDouble(), 'f', 5));
        }

        painter->restore();
    }
}

QSize GoodsDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    return { 88, 68 };
}
