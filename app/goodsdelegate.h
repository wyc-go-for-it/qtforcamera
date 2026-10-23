#ifndef GOODSDELEGATE_H
#define GOODSDELEGATE_H

#include <QPainterPath>
#include <QStyledItemDelegate>

class GoodsDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    explicit GoodsDelegate(QObject* parent = nullptr);
    virtual ~GoodsDelegate() { }

protected:
    virtual void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override final;
    virtual QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

private:
    mutable QPainterPath m_path;
};

#endif // GOODSDELEGATE_H
