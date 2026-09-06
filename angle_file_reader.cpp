#include "angle_file_reader.h"
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <map>
#include <stdexcept>

std::vector<double> readAnglesFromSicsFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        throw std::runtime_error("Failed to open angle file: " + path.toStdString());

    static const QRegularExpression lineRe(R"(^DINGO_(\d+)\s.*abr\s*=\s*(-?[0-9]*\.?[0-9]+))");

    std::map<int, double> anglesByIndex;
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        QRegularExpressionMatch m = lineRe.match(line);
        if (!m.hasMatch())
            continue;
        anglesByIndex[m.captured(1).toInt()] = m.captured(2).toDouble();
    }

    if (anglesByIndex.empty())
        throw std::runtime_error("No angle entries found in " + path.toStdString());

    const int n = anglesByIndex.rbegin()->first + 1;
    if (static_cast<int>(anglesByIndex.size()) != n)
        throw std::runtime_error("Angle file has gaps in projection indices (expected " +
                                  std::to_string(n) + " contiguous entries from 0, found " +
                                  std::to_string(anglesByIndex.size()) + ")");

    std::vector<double> angles(n);
    for (const auto& [index, angle] : anglesByIndex)
        angles[index] = angle;
    return angles;
}
