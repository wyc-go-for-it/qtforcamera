#ifndef GOODSMODEL_H
#define GOODSMODEL_H

#include "productdatabase.h"
#include "qobject.h"

#include <QAbstractListModel>
#include <QList>

template <typename T>
class BasetModel : public QAbstractListModel {
public:
    explicit BasetModel(QObject* parent) {

    };

public:
    Qt::ItemFlags flags(const QModelIndex& index) const override
    {
        if (!index.isValid())
            return {};

        return QAbstractListModel::flags(index) | Qt::ItemIsSelectable;
    }

    QModelIndex sibling(int row, int column, const QModelIndex& idx) const override final
    {
        if (!idx.isValid() || column != 0 || row >= lst.count() || row < 0)
            return QModelIndex();

        return createIndex(row, 0);
    }

    QModelIndex index(int row, int column = 0, const QModelIndex& parent = QModelIndex()) const override
    {
        return QAbstractListModel::index(row, column, parent);
    }

    int rowCount(const QModelIndex& parent = QModelIndex()) const override final
    {
        return parent.isValid() ? 0 : lst.count();
    }

    int addRows(const QList<T>& val)
    {
        if (val.isEmpty()) {
            return lst.count();
        }

        beginInsertRows(QModelIndex(), lst.size(), lst.size() + val.size() - 1);

        lst.append(val);

        endInsertRows();

        return lst.count();
    }

    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override
    {
        if (count <= 0 || row < 0 || (row + count) > rowCount(parent))
            return false;

        beginRemoveRows(parent, row, row + count - 1);

        const auto it = lst.begin() + row;
        lst.erase(it, it + count);

        endRemoveRows();

        return true;
    }

    void clear()
    {
        removeRows(0, lst.size());
    }

protected:
    QList<T> lst;
};

class GoodsModel : public BasetModel<ProductRecord> {
public:
    explicit GoodsModel(QObject* parent)
        : BasetModel(parent)
    {
        loadData();
    };

public:
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override
    {
        if (index.row() < 0 || index.row() >= lst.size())
            return {};

        if (role == Qt::DisplayRole) {
            return QString::fromStdString(lst.at(index.row()).name);
        } else if (role == Qt::UserRole + 1) {
            return lst.at(index.row()).id;
        } else if (role == Qt::UserRole + 2) {
            return QString::fromStdString(lst.at(index.row()).barcode);
        }

        return {};
    }

private:
    void loadData();
};

class RecogModel : public BasetModel<SearchResult> {
public:
    explicit RecogModel(QObject* parent)
        : BasetModel(parent) {

        };

public:
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override
    {
        if (index.row() < 0 || index.row() >= lst.size())
            return {};

        if (role == Qt::DisplayRole) {
            return QString::fromStdString(lst.at(index.row()).name);
        } else if (role == Qt::UserRole + 1) {
            return lst.at(index.row()).id;
        } else if (role == Qt::UserRole + 9) {
            return lst.at(index.row()).similarity;
        }

        return {};
    }
};

#endif // GOODSMODEL_H
