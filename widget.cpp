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
  ui->doubleSpinBox->setDecimals(2);
}

Widget::~Widget() {
  delete ui;
  delete customPlotTitle;
}

void Widget::on_p0Button1_pressed() {
  QString strImageFile = QFileDialog::getOpenFileName(
      0, "Open", "/home", tr("Images(*.png *jpg *bmp)"));
  if (strImageFile.isEmpty())
    return;
  image0 = QImage(strImageFile);
  ui->infLabel->setText(QString("打开文件:" + strImageFile));
  image0Szie = image0.size();
  ui->label_20->setText(QString("文件信息: %1 X %2 ")
                            .arg(image0Szie.width())
                            .arg(image0Szie.height()));
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
  QString filename =
      QFileDialog::getSaveFileName(this, tr("Save Image"), "optput.jpg",
                                   tr("Images (*.png *.bmp *.jpg)")); //选择路径
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

void Widget::on_p0Button1_6_clicked() {
  imMake.makeQtsu(image0, &image2);
  ui->infLabel->setText(QString("Otsu’s 二值化计算完毕，结果在图像3"));
}

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

void Widget::on_BTSlider_15_valueChanged(int value) {
  double inf;
  if (value <= 100) {
    inf = value / 100.0;
    ui->doubleSpinBox->setValue(inf);
  } else {
    inf = ((value - 100) * 9 + 100) / 100.0;
    ui->doubleSpinBox->setValue(inf);
  }
  image0ToSzie = QSize(qint64(image0Szie.width() * inf) < 1
                           ? 1
                           : qint64(image0Szie.width() * inf),
                       qint64(image0Szie.height() * inf) < 1
                           ? 1
                           : qint64(image0Szie.height() * inf));
  ui->label_21->setText(QString("缩放后大小: %1 X %2")
                            .arg(image0ToSzie.width())
                            .arg(image0ToSzie.height()));
}

void Widget::on_doubleSpinBox_valueChanged(double arg1) {
  double inf;
  if (arg1 <= 1) {
    inf = arg1 * 100;
    ui->BTSlider_15->setValue(inf);
  } else {
    inf = (arg1 * 100 - 100) / 9.0 + 100;
    ui->BTSlider_15->setValue(inf);
  }
  image0ToSzie = QSize(qint64(image0Szie.width() * arg1) < 1
                           ? 1
                           : qint64(image0Szie.width() * arg1),
                       qint64(image0Szie.height() * arg1) < 1
                           ? 1
                           : qint64(image0Szie.height() * arg1));
  ui->label_21->setText(QString("缩放后大小: %1 X %2")
                            .arg(image0ToSzie.width())
                            .arg(image0ToSzie.height()));
}

void Widget::on_BTButton_2_clicked() {
  if (image0.isNull())
    return;
  image1 = image0.scaled(image0ToSzie, Qt::KeepAspectRatio,
                         Qt::SmoothTransformation);
  ui->infLabel->setText(
      QString("图片缩放已计算完毕,结果在图像2,分辨率为 %1 X %2")
          .arg(image1.size().width())
          .arg(image1.size().height()));
}

void Widget::on_p0Button1_16_clicked() {
  imMake.makeFrequencyDfilter(image0, &image2, &image3, ui->spinBox_15->value(),
                              0, 1);
  ui->infLabel->setText(QString(
      "高斯滤波已计算完毕,高斯低通滤波结果在图像3，高斯高通滤波结果在图像4"));
}

void Widget::on_p0Button1_17_clicked() {
  imMake.makeFrequencyDfilter(image0, &image2, &image3, ui->spinBox_16->value(),
                              0, 0);
  ui->infLabel->setText(QString(
      "理想滤波已计算完毕,理想低通滤波结果在图像3，理想高通滤波结果在图像4"));
}

void Widget::on_p0Button1_18_clicked() {
  imMake.makeFrequencyDfilter(image0, &image2, &image3, ui->spinBox_17->value(),
                              ui->spinBox_18->value(), 2);
  ui->infLabel->setText(
      QString("巴特沃斯滤波已计算完毕,"
              "巴特沃斯低通滤波结果在图像3，巴特沃斯高通滤波结果在图像4"));
}

void Widget::on_doubleSpinBox_2_valueChanged(double arg1) {
  ui->BTSlider_19->setValue(arg1 * 100);
}

void Widget::on_BTSlider_20_valueChanged(int value) {
  ui->doubleSpinBox_2->setValue(value / 100.0);
}

void Widget::on_p0Button1_19_clicked() {
  imMake.makeUSM(image0, &image2, ui->doubleSpinBox_2->value(),
                 ui->spinBox_19->value());
  ui->infLabel->setText(QString("USM锐化已计算完毕,"
                                "计算结果在图像3"));
}

void Widget::on_p0Button1_20_clicked() {
  imMake.makeLaplacianSharpen(image0, &image2);
  ui->infLabel->setText(QString("拉普拉斯算子图像锐化已计算完毕,"
                                "计算结果在图像3"));
}

void Widget::on_p0Button1_21_clicked() {

  if (ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
    imMake.makeAdaptiveMedianFilter(image1, &image2);
  else
    imMake.makeAdaptiveMedianFilter(image2, &image3);
  ui->infLabel->setText(
      QString("自适应中值滤波变换计算完毕，结果在图像") +
      QString(((ui->buttonGroup_2->checkedButton() == ui->radioButton_3)
                   ? "3"
                   : "4")));
}

void Widget::on_p0Button1_22_clicked() {
  imMake.makeHomomorphicFilter(
      image0, &image2, ui->spinBox_20->value(), ui->doubleSpinBox_4->value(),
      ui->doubleSpinBox_5->value(), ui->doubleSpinBox_6->value());
}
