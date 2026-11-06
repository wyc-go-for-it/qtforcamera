#pragma once

#include <QDialog>

class QLabel;
class QPushButton;

class FilePickerDialog : public QDialog {
    Q_OBJECT
public:
    explicit FilePickerDialog(QWidget* parent = nullptr);
    ~FilePickerDialog() override;

    QString selectedPath() const { return selectedPath_; }
    QStringList selectedPaths() const { return selectedPaths_; }
    QString selectedDir() const { return selectedDir_; }

private slots:
    void onPickSingleFile();
    void onPickMultipleFiles();
    void onPickDirectory();

private:
    QString defaultDir() const;

    QLabel* label_ = nullptr;
    QString selectedPath_;
    QStringList selectedPaths_;
    QString selectedDir_;
    QString lastDir_ = ".";
};
