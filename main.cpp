#include "widget.h"

#include <QApplication>

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  /* 设置中文字体 */
  a.setFont(QFont("Microsoft Yahei", 9));

  /* 设置中文编码 */
#if (QT_VERSION <= QT_VERSION_CHECK(5, 0, 0))
#if _MSC_VER
  QTextCodec *codec = QTextCodec::codecForName("gbk");
#else
  QTextCodec *codec = QTextCodec::codecForName("utf-8");
#endif
  QTextCodec::setCodecForLocale(codec);
  QTextCodec::setCodecForCStrings(codec);
  QTextCodec::setCodecForTr(codec);
#else
  QTextCodec *codec = QTextCodec::codecForName("utf-8");
  QTextCodec::setCodecForLocale(codec);
#endif

#if 0 /* not need css file */
  QFile qssFile(":/resource/qss/style.qss");
  qssFile.open(QFile::ReadOnly); //以只读方式打开
  if (qssFile.isOpen()) {
    QString qss = QLatin1String(qssFile.readAll());
    qApp->setStyleSheet(qss);
    qssFile.close();
  } else {
    qDebug() << "无法打开qss资源文件";
  }
#endif
  Widget w;
  w.show();
  return a.exec();
}
