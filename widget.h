#ifndef WIDGET_H
#define WIDGET_H

#include <QCursor>
#include <QFileDialog>
#include <QImage>
#include <QWidget>
#include <qcustomplot/qcustomplot.h>

#include "imagemake.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget {
  Q_OBJECT

public:
  QTimer timerOfUpImage;
  QImage image0;
  QImage image1;
  QImage image2;
  QImage image3;
  QCustomPlot customPlot;
  QCPTextElement *customPlotTitle;
  imageMake imMake;
  QSize image0Szie = QSize(1920, 1080);
  QSize image0ToSzie = QSize(1920, 1080);

  QMenu image0Menu;
  QMenu image1Menu;
  QMenu image2Menu;
  QMenu image3Menu;

  void saveImage(QImage image);
  //  void updateImage0(void);
  //  void updateImage1(void);
  //  void updateImage2(void);
  //  void updateImage3(void);

  virtual bool eventFilter(QObject *obj, QEvent *event) override;

public slots:
  void timeOfUp();

public:
  Widget(QWidget *parent = nullptr);
  ~Widget();

private slots:
  void on_p0Button1_pressed();

  void on_p0Button1_2_clicked();

  void on_p0Button1_3_clicked();

  void on_BTButton_clicked();

  void on_BTSlider_2_valueChanged(int value);

  void on_spinBox_2_valueChanged(int arg1);

  void on_p0Button1_5_clicked();

  void on_p0Button1_6_clicked();

  void on_spinBox_8_valueChanged(int arg1);

  void on_BTSlider_8_valueChanged(int value);

  void on_p0Button1_10_clicked();

  void on_spinBox_9_valueChanged(int arg1);

  void on_BTSlider_9_valueChanged(int value);

  void on_p0Button1_11_clicked();

  void on_p0Button1_12_clicked();

  void on_p0Button1_13_clicked();

  void on_p0Button1_14_clicked();

  void on_p0Button1_15_clicked();

  void on_p0Button1_4_clicked();

  void on_BTSlider_4_valueChanged(int value);

  void on_spinBox_4_valueChanged(int arg1);

  void on_spinBox_5_valueChanged(int arg1);

  void on_BTSlider_5_valueChanged(int value);

  void on_p0Button1_7_clicked();

  void on_spinBox_7_valueChanged(int arg1);

  void on_BTSlider_7_valueChanged(int value);

  void on_spinBox_6_valueChanged(int arg1);

  void on_BTSlider_6_valueChanged(int value);

  void on_p0Button1_8_clicked();

  void on_p0Button1_9_clicked();

  void on_p0Button3_clicked();

  void on_p0Button3_2_clicked();

  void on_p0Button3_3_clicked();

  void on_p0Button3_4_clicked();

  void on_BTSlider_15_valueChanged(int value);

  void on_doubleSpinBox_valueChanged(double arg1);

  void on_BTButton_2_clicked();

  void on_p0Button1_16_clicked();

  void on_p0Button1_17_clicked();

  void on_p0Button1_18_clicked();

  void on_doubleSpinBox_2_valueChanged(double arg1);

  void on_BTSlider_20_valueChanged(int value);

  void on_p0Button1_19_clicked();

  void on_p0Button1_20_clicked();

  void on_p0Button1_21_clicked();

  void on_p0Button1_22_clicked();

  void on_BTButton_3_clicked();

  void on_BTButton_4_clicked();

  void on_saveAc1_triggered();

  void on_saveAc2_triggered();

  void on_saveAc3_triggered();

  void on_saveAc4_triggered();

  void on_gAc1_triggered();

  void on_gAc2_triggered();

  void on_gAc3_triggered();

  void on_gAc4_triggered();

  void on_show1_triggered();

  void on_show2_triggered();

  void on_show3_triggered();

  void on_show4_triggered();

  void on_openIm_triggered();

  void on_p0Button1_23_clicked();

  void on_p0Button1_24_clicked();

  void on_p0Button1_25_clicked();

  void on_p0Button1_26_clicked();

private:
  Ui::Widget *ui;
};
#endif // WIDGET_H
