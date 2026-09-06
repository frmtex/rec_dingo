#ifndef ANGLE_FILE_READER_H
#define ANGLE_FILE_READER_H
#include <QString>
#include <vector>

// Reads real rotation-axis encoder angles (degrees) recorded during a scan, as written by the
// SICS instrument control software to a log like sics_out.txt - one line per frame, e.g.
// "DINGO_000000\t time:2025-11-21 13:19:57.237787\tabr = 0.007601". Flat-field (DINGO_ob_*) and
// dark-field (DINGO_di_*) lines have no angle and are skipped automatically, since \d+ right
// after "DINGO_" doesn't match their non-numeric suffix.
//
// Returns angles indexed by projection number (angles[a] is the encoder reading recorded for
// DINGO_%06d == a), covering every index from 0 up to the highest one found. Throws
// std::runtime_error if the file can't be opened, no angle lines are found, or the projection
// indices have gaps.
std::vector<double> readAnglesFromSicsFile(const QString& path);

#endif // ANGLE_FILE_READER_H
