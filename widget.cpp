#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget) {
  ui->setupUi(this);
  ui->showLayout->addWidget(&customPlot);
  connect(&timerOfUpImage, SIGNAL(timeout()), this, SLOT(timeOfUp()));
  timerOfUpImage.start(100);
  customPlot.addGraph();
  customPlot.plotLayout()->insertRow(0);
  customPlotTitle = new QCPTextElement(&customPlot, tr("Grayscale histogram"));
  customPlot.plotLayout()->addElement(0, 0, customPlotTitle);
  customPlot.graph(0)->setLineStyle(QCPGraph::lsImpulse);
  ui->infLabel->setText(QString("启动完成"));
}

Widget::~Widget() {
  delete ui;
  delete customPlotTitle;
}

void Widget::on_p0Button1_pressed() {
  QString strImageFile = QFileDialog::getOpenFileName(0, "Open", "/home",
                                                      tr("Images(*.png *jpg)"));
  if (strImageFile.isEmpty())
    return;
  image0 = QImage(strImageFile);
  ui->infLabel->setText(QString("打开文件:" + strImageFile));
}

void Widget::on_p0Button1_2_clicked() {
  imMake.makeHistogram(image0, &customPlot, &image1);
  ui->infLabel->setText(
      QString("灰度图，直方图计算完毕，结果在图像2，直方图在右上角图表"));
}

void Widget::on_p0Button1_3_clicked() {
  imMake.makeHistogramAveraging(image1, &customPlot, &image2);
  ui->infLabel->setText(
      QString("直方图均衡化计算完毕，结果在图像3，均值后直方图在右上角图表"));
}
void Widget::saveImage(QImage image) {
  cv::Mat mat0 = imMake.QImageToMat(image);
  QString filename = QFileDialog::getSaveFileName(
      this, tr("Save Image"), "", tr("Images (*.png *.bmp *.jpg)")); //选择路径
  if (filename.isEmpty())
    return;
  std::string fileAsSave = filename.toStdString();
  imwrite(fileAsSave, mat0);
}

void Widget::timeOfUp() {
  QImage tempi;
  if (!image0.isNull()) {
    tempi = image0.scaled(ui->image0showLabel->width(),
                          ui->image0showLabel->height(),
                          Qt::AspectRatioMode::KeepAspectRatio);
    ui->image0showLabel->setPixmap(QPixmap::fromImage(tempi));
  }
  if (!image1.isNull()) {
    tempi = image1.scaled(ui->image1showLabel->width(),
                          ui->image1showLabel->height(),
                          Qt::AspectRatioMode::KeepAspectRatio);
    ui->image1showLabel->setPixmap(QPixmap::fromImage(tempi));
  }
  if (!image2.isNull()) {
    tempi = image2.scaled(ui->image2showLabel->width(),
                          ui->image2showLabel->height(),
                          Qt::AspectRatioMode::KeepAspectRatio);
    ui->image2showLabel->setPixmap(QPixmap::fromImage(tempi));
  }
  if (!image3.isNull()) {
    tempi = image3.scaled(ui->image3showLabel->width(),
                          ui->image3showLabel->height(),
                          Qt::AspectRatioMode::KeepAspectRatio);
    ui->image3showLabel->setPixmap(QPixmap::fromImage(tempi));
  }
}

void Widget::on_BTButton_clicked() {
  imMake.makeBinarization(image0, &image2, (quint8)ui->spinBox->value());
  ui->infLabel->setText(QString("二值化计算完毕，结果在图像3"));
}

void Widget::on_BTSlider_2_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_2->setValue(value);
  }
  ui->spinBox_2->setValue(value);
}

void Widget::on_spinBox_2_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_2->setValue(arg1);
  }
  ui->BTSlider_2->setValue(arg1);
}

void Widget::on_p0Button1_5_clicked() {
  imMake.makeAdaptiveThreshold(image0, &image2, &image3,
                               (quint8)ui->spinBox_2->value(),
                               (quint8)ui->spinBox_3->value());
  ui->infLabel->setText(QString("区域自适应二值化计算完毕，领域的算术平均值结果"
                                "在图像3，领域的高斯均值结果在图像4"));
}

void Widget::on_p0Button1_6_clicked() { imMake.makeQtsu(image0, &image2); }

void Widget::on_spinBox_8_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_8->setValue(arg1);
  }
  ui->BTSlider_8->setValue(arg1);
}

void Widget::on_BTSlider_8_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_8->setValue(value);
  }
  ui->spinBox_8->setValue(value);
}

void Widget::on_p0Button1_10_clicked() {
  imMake.makeSobel(image0, &image2, ui->spinBox_8->value());
  ui->infLabel->setText(QString("Sobel算子计算完毕，结果在图像3"));
}

void Widget::on_spinBox_9_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_9->setValue(arg1);
  }
  ui->BTSlider_9->setValue(arg1);
}

void Widget::on_BTSlider_9_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_9->setValue(value);
  }
  ui->spinBox_9->setValue(value);
}

void Widget::on_p0Button1_11_clicked() {
  imMake.makeLaplace(image0, &image2, ui->spinBox_9->value());
  ui->infLabel->setText(QString("Laplace算子计算完毕，结果在图像3"));
}

void Widget::on_p0Button1_12_clicked() {
  imMake.makeCanny(image0, &image2, ui->spinBox_10->value());
  ui->infLabel->setText(QString("Canny算子计算完毕，结果在图像3"));
}

void Widget::on_p0Button1_13_clicked() {
  imMake.makeScharr(image0, &image2);
  ui->infLabel->setText(QString("Scharr算子计算完毕，结果在图像3"));
}

void Widget::on_p0Button1_14_clicked() {
  imMake.makeHoughLines(image2, &image3, ui->spinBox_11->value());
  ui->infLabel->setText(QString("标准霍夫线变换计算完毕，结果在图像4"));
}

void Widget::on_p0Button1_15_clicked() {
  imMake.makeHoughLinesP(image2, &image3, ui->spinBox_12->value());
  ui->infLabel->setText(QString("统计概率霍夫线变换计算完毕，结果在图像4"));
}

void Widget::on_p0Button1_4_clicked() {
  if (ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
    imMake.makeMeanFilter(image1, &image2, ui->spinBox_4->value());
  else
    imMake.makeMeanFilter(image2, &image3, ui->spinBox_4->value());
  ui->infLabel->setText(
      QString("均值滤波变换计算完毕，结果在图像") +
      QString(((ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
                   ? "3"
                   : "4")));
}

void Widget::on_BTSlider_4_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_4->setValue(value);
  }
  ui->spinBox_4->setValue(value);
}

void Widget::on_spinBox_4_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_4->setValue(arg1);
  }
  ui->BTSlider_4->setValue(arg1);
}

void Widget::on_spinBox_5_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_5->setValue(arg1);
  }
  ui->BTSlider_5->setValue(arg1);
}

void Widget::on_BTSlider_5_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_5->setValue(value);
  }
  ui->spinBox_5->setValue(value);
}

void Widget::on_p0Button1_7_clicked() {
  if (ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
    imMake.makeGaussianFilter(image1, &image2, ui->spinBox_5->value());
  else
    imMake.makeGaussianFilter(image2, &image3, ui->spinBox_5->value());
  ui->infLabel->setText(
      QString("高斯滤波变换计算完毕，结果在图像") +
      QString(((ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
                   ? "3"
                   : "4")));
}

void Widget::on_spinBox_7_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_7->setValue(arg1);
  }
  ui->BTSlider_7->setValue(arg1);
}

void Widget::on_BTSlider_7_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_7->setValue(value);
  }
  ui->spinBox_7->setValue(value);
}

void Widget::on_spinBox_6_valueChanged(int arg1) {
  if (arg1 % 2 == 0) {
    arg1 = arg1 + 1;
    ui->spinBox_6->setValue(arg1);
  }
  ui->BTSlider_6->setValue(arg1);
}

void Widget::on_BTSlider_6_valueChanged(int value) {
  if (value % 2 == 0) {
    value = value + 1;
    ui->BTSlider_6->setValue(value);
  }
  ui->spinBox_6->setValue(value);
}

void Widget::on_p0Button1_8_clicked() {
  if (ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
    imMake.makeMedianFilter(image1, &image2, ui->spinBox_7->value());
  else
    imMake.makeMedianFilter(image2, &image3, ui->spinBox_7->value());
  ui->infLabel->setText(
      QString("中值滤波变换计算完毕，结果在图像") +
      QString(((ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
                   ? "3"
                   : "4")));
}

void Widget::on_p0Button1_9_clicked() {
  if (ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
    imMake.makeBilateralFilter(image1, &image2, ui->spinBox_6->value(),
                               ui->spinBox_13->value(),
                               ui->spinBox_14->value());
  else
    imMake.makeBilateralFilter(image2, &image3, ui->spinBox_6->value(),
                               ui->spinBox_13->value(),
                               ui->spinBox_14->value());
  ui->infLabel->setText(
      QString("双边滤波变换计算完毕，结果在图像") +
      QString(((ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
                   ? "3"
                   : "4")));
}

void Widget::on_p0Button3_clicked() { saveImage(image0); }

void Widget::on_p0Button3_2_clicked() { saveImage(image1); }

void Widget::on_p0Button3_3_clicked() { saveImage(image2); }

void Widget::on_p0Button3_4_clicked() { saveImage(image3); }
