#include "filepickerdialog.h"
#include "qdebug.h"

#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStandardPaths>
#include <QVBoxLayout>

FilePickerDialog::FilePickerDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("选择文件");
    setModal(true); // 默认模态，exec() 时也模态
    resize(520, 260);

    auto* layout = new QVBoxLayout(this);

    auto* btnFile = new QPushButton("选择单个图片", this);
    auto* btnFiles = new QPushButton("选择多个图片", this);
    auto* btnDir = new QPushButton("选择文件夹", this);

    label_ = new QLabel("尚未选择", this);
    label_->setWordWrap(true);

    layout->addWidget(btnFile);
    layout->addWidget(btnFiles);
    layout->addWidget(btnDir);
    layout->addWidget(label_);

    // 标准 OK / Cancel 按钮
    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    connect(btnFile, &QPushButton::clicked, this, &FilePickerDialog::onPickSingleFile);
    connect(btnFiles, &QPushButton::clicked, this, &FilePickerDialog::onPickMultipleFiles);
    connect(btnDir, &QPushButton::clicked, this, &FilePickerDialog::onPickDirectory);
}

FilePickerDialog::~FilePickerDialog() = default;

QString FilePickerDialog::defaultDir() const
{
    if (!lastDir_.isEmpty())
        return lastDir_;
    return QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
}

void FilePickerDialog::onPickSingleFile()
{
    QString path = QFileDialog::getOpenFileName(
        this, "选择一张图片", defaultDir(),
        "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)");
    if (path.isEmpty())
        return;

    selectedPath_ = path;
    lastDir_ = QFileInfo(path).absolutePath();
    label_->setText("已选择:\n" + path);
}

void FilePickerDialog::onPickMultipleFiles()
{
    QStringList paths = QFileDialog::getOpenFileNames(
        this, "选择多张图片", defaultDir(),
        "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)");
    if (paths.isEmpty())
        return;

    selectedPaths_ = paths;
    lastDir_ = QFileInfo(paths.first()).absolutePath();
    label_->setText(QString("已选择 %1 个文件").arg(paths.size()));
}

void FilePickerDialog::onPickDirectory()
{
    QString dir = QFileDialog::getExistingDirectory(
        this, "选择文件夹", defaultDir(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (dir.isEmpty())
        return;

    selectedDir_ = dir;
    lastDir_ = dir;
    label_->setText("目录:\n" + dir);
}
