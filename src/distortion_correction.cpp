#include "distortion_correction.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace {

// ---------------------------------------------------------------- image input

// Minimal FITS reader: the primary image only, 2-D, BITPIX 8/16/32/-32/-64, BZERO/BSCALE honoured.
// Rows are returned in file order (FITS bottom-up files are fixed with the "flip" option).
cv::Mat readFits(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        throw std::runtime_error("cannot open " + path.toStdString());

    int bitpix = 0, naxis = 0, n1 = 0, n2 = 0;
    double bzero = 0.0, bscale = 1.0;
    bool ended = false;
    while (!ended) {
        QByteArray block = f.read(2880);
        if (block.size() < 2880)
            throw std::runtime_error("truncated FITS header in " + path.toStdString());
        for (int i = 0; i < 2880; i += 80) {
            const QByteArray card = block.mid(i, 80);
            const QString key = QString::fromLatin1(card.left(8)).trimmed();
            if (key == "END") { ended = true; break; }
            if (card.size() < 10 || card[8] != '=')
                continue;
            const QString value = QString::fromLatin1(card.mid(10)).split('/').first().trimmed();
            if (key == "BITPIX") bitpix = value.toInt();
            else if (key == "NAXIS") naxis = value.toInt();
            else if (key == "NAXIS1") n1 = value.toInt();
            else if (key == "NAXIS2") n2 = value.toInt();
            else if (key == "BZERO") bzero = value.toDouble();
            else if (key == "BSCALE") bscale = value.toDouble();
        }
    }
    if (naxis != 2 || n1 <= 0 || n2 <= 0)
        throw std::runtime_error("only 2-D FITS images are supported: " + path.toStdString());

    const int bytes = std::abs(bitpix) / 8;
    if (bytes == 0 || (bitpix != 8 && bitpix != 16 && bitpix != 32 && bitpix != -32 && bitpix != -64))
        throw std::runtime_error("unsupported FITS BITPIX in " + path.toStdString());
    const QByteArray raw = f.read(static_cast<qint64>(n1) * n2 * bytes);
    if (raw.size() < static_cast<qint64>(n1) * n2 * bytes)
        throw std::runtime_error("truncated FITS data in " + path.toStdString());

    cv::Mat out(n2, n1, CV_32FC1);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.constData());
    for (int y = 0; y < n2; ++y) {
        float* dst = out.ptr<float>(y);
        for (int x = 0; x < n1; ++x, p += bytes) {
            double v = 0.0;
            switch (bitpix) {
            case 8: v = p[0]; break;
            case 16: v = static_cast<int16_t>((p[0] << 8) | p[1]); break;
            case 32: v = static_cast<int32_t>((uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]); break;
            case -32: {
                uint32_t u = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
                float fv; std::memcpy(&fv, &u, 4); v = fv; break;
            }
            default: {
                uint64_t u = 0;
                for (int k = 0; k < 8; ++k) u = (u << 8) | p[k];
                double dv; std::memcpy(&dv, &u, 8); v = dv; break;
            }
            }
            dst[x] = static_cast<float>(bzero + bscale * v);
        }
    }
    return out;
}

cv::Mat toGrayFloat(const cv::Mat& img)
{
    cv::Mat g;
    if (img.channels() == 1) g = img;
    else cv::cvtColor(img, g, img.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    cv::Mat f;
    g.convertTo(f, CV_32FC1);
    return f;
}

// ------------------------------------------------------------ grid detection

// Grid pitch (pixels per cell) from the strongest periodic component of the central part of the
// flattened image. Only used to size the filters below; the final pitch is measured from the
// detected crossings.
double estimatePitch(const cv::Mat& flat)
{
    const int n = std::min({2048, flat.cols, flat.rows});
    cv::Mat crop = flat(cv::Rect((flat.cols - n) / 2, (flat.rows - n) / 2, n, n)).clone();
    crop -= cv::mean(crop)[0];
    cv::Mat win;
    cv::createHanningWindow(win, crop.size(), CV_32F);
    cv::multiply(crop, win, crop);
    cv::Mat spec;
    cv::dft(crop, spec, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat planes[2];
    cv::split(spec, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    // Period between 8 and 400 px -> frequency index between n/400 and n/8.
    const double kMin = n / 400.0, kMax = n / 8.0;
    double best = 0.0;
    int bx = 0, by = 0;
    for (int y = 0; y < n; ++y) {
        const int fy = y <= n / 2 ? y : y - n;
        const float* row = mag.ptr<float>(y);
        for (int x = 0; x <= n / 2; ++x) {
            const double k = std::hypot(double(x), double(fy));
            if (k < kMin || k > kMax) continue;
            if (row[x] > best) { best = row[x]; bx = x; by = y; }
        }
    }
    if (best <= 0.0)
        throw std::runtime_error("no periodic structure found - is this an image of graph paper?");
    // Parabolic refinement of the peak position along the dominant axis.
    auto at = [&](int x, int y) { return double(mag.at<float>((y + n) % n, std::abs(x))); };
    auto refine = [&](double c, double m, double p) {
        const double d = m - 2 * c + p;
        return d == 0.0 ? 0.0 : 0.5 * (m - p) / d;
    };
    const double fx = bx + refine(at(bx, by), at(bx - 1, by), at(bx + 1, by));
    const int sy = by <= n / 2 ? by : by - n;
    const double fyy = sy + refine(at(bx, by), at(bx, by - 1), at(bx, by + 1));
    return n / std::hypot(fx, fyy);
}

struct Crossing { double x, y; };

// Sub-pixel crossings of dark grid lines. `pitch` sizes the filters.
std::vector<Crossing> detectCrossings(const cv::Mat& flat, double pitch)
{
    const double sigLine = std::max(1.0, 0.045 * pitch);
    const double sigAlong = std::max(2.0, 0.13 * pitch);
    const int nms = std::max(2, int(std::lround(0.22 * pitch)));
    const int half = std::max(2, int(std::lround(0.09 * pitch)));

    cv::Mat g;
    cv::GaussianBlur(flat, g, cv::Size(0, 0), sigLine);
    cv::Mat lxx, lyy;
    cv::Sobel(g, lxx, CV_32F, 2, 0, 3);
    cv::Sobel(g, lyy, CV_32F, 0, 2, 3);
    g.release();
    cv::max(lxx, 0.0f, lxx); // dark vertical lines
    cv::max(lyy, 0.0f, lyy); // dark horizontal lines
    cv::GaussianBlur(lxx, lxx, cv::Size(0, 0), 0.01, sigAlong); // average along the line direction
    cv::GaussianBlur(lyy, lyy, cv::Size(0, 0), sigAlong, 0.01);
    cv::Mat P;
    cv::multiply(lxx, lyy, P); // large only where a vertical and a horizontal line cross
    lxx.release();
    lyy.release();

    cv::Mat mx;
    cv::dilate(P, mx, cv::Mat::ones(2 * nms + 1, 2 * nms + 1, CV_8U));
    std::vector<cv::Point> cand;
    std::vector<float> val;
    for (int y = half; y < P.rows - half; ++y) {
        const float* pr = P.ptr<float>(y);
        const float* mr = mx.ptr<float>(y);
        for (int x = half; x < P.cols - half; ++x)
            if (pr[x] > 0.0f && pr[x] == mr[x]) { cand.emplace_back(x, y); val.push_back(pr[x]); }
    }
    if (cand.size() < 50)
        throw std::runtime_error("no grid crossings found - check the grid images (focus, contrast, "
                                 "and that the paper fills a good part of the frame)");

    std::vector<float> lg(val.size());
    for (size_t i = 0; i < val.size(); ++i) lg[i] = std::log10(val[i]);

    // Real crossings and noise maxima separate in log-strength, but there can be more than two
    // clusters (noise, weak secondary structure, the crossings). Split with Otsu's method and
    // keep splitting the upper part until no more points remain than the image can hold at this
    // pitch (one crossing per pitch^2 of area at most).
    auto otsu = [](const std::vector<float>& v, float& outThr) {
        const auto mm = std::minmax_element(v.begin(), v.end());
        const float lo = *mm.first, hi = *mm.second;
        if (hi - lo < 1e-6f) return false;
        std::vector<double> hist(256, 0.0);
        for (float x : v) hist[std::min(255, int((x - lo) / (hi - lo) * 256.0))]++;
        const double total = double(v.size());
        double sumAll = 0;
        for (int i = 0; i < 256; ++i) sumAll += i * hist[i];
        double wB = 0, sumB = 0, bestVar = -1;
        int bestT = 0;
        for (int t = 0; t < 256; ++t) {
            wB += hist[t];
            if (wB == 0) continue;
            const double wF = total - wB;
            if (wF == 0) break;
            sumB += t * hist[t];
            const double mB = sumB / wB, mF = (sumAll - sumB) / wF;
            const double var = wB * wF * (mB - mF) * (mB - mF);
            if (var > bestVar) { bestVar = var; bestT = t; }
        }
        outThr = lo + (bestT + 1) * (hi - lo) / 256.0f;
        return true;
    };
    const double maxCrossings = 1.1 * double(flat.cols) * flat.rows / (pitch * pitch);
    float thr = -1e30f;
    std::vector<float> upper = lg;
    for (int round = 0; round < 5 && double(upper.size()) > maxCrossings; ++round) {
        float t;
        if (!otsu(upper, t)) break;
        thr = t;
        std::vector<float> next;
        for (float x : upper) if (x >= thr) next.push_back(x);
        if (next.size() < 50) break; // a split that leaves nothing useful - keep the previous one
        upper.swap(next);
    }

    std::vector<Crossing> out;
    for (size_t i = 0; i < cand.size(); ++i) {
        if (lg[i] < thr) continue;
        const int cx = cand[i].x, cy = cand[i].y;
        double wmax = 0;
        for (int dy = -half; dy <= half; ++dy)
            for (int dx = -half; dx <= half; ++dx)
                wmax = std::max(wmax, double(P.at<float>(cy + dy, cx + dx)));
        double sw = 0, sx = 0, sy = 0;
        for (int dy = -half; dy <= half; ++dy)
            for (int dx = -half; dx <= half; ++dx) {
                const double w = std::max(double(P.at<float>(cy + dy, cx + dx)) - 0.3 * wmax, 0.0);
                sw += w; sx += w * dx; sy += w * dy;
            }
        if (sw > 0) out.push_back({cx + sx / sw, cy + sy / sw});
    }
    return out;
}

// ------------------------------------------------------------- lattice index

struct Lattice { std::vector<cv::Point2d> ij; std::vector<cv::Point2d> xy; double pitch = 0; };

// Assigns every crossing integer lattice coordinates by walking outwards from the one nearest the
// image centre, predicting each neighbour from the local spacing vectors (so slow rotation and
// perspective don't matter) - and drops anything that doesn't sit where a neighbour should be.
Lattice indexLattice(const std::vector<Crossing>& pts, cv::Size frame, double pitch)
{
    // Spatial hash.
    const double cell = pitch;
    auto key = [](int cx, int cy) { return (int64_t(cx) << 32) ^ uint32_t(cy); };
    std::unordered_map<int64_t, std::vector<int>> grid;
    for (size_t i = 0; i < pts.size(); ++i)
        grid[key(int(std::floor(pts[i].x / cell)), int(std::floor(pts[i].y / cell)))].push_back(int(i));
    auto nearest = [&](cv::Point2d q, double& dist) {
        const int cx = int(std::floor(q.x / cell)), cy = int(std::floor(q.y / cell));
        int bestI = -1; double bestD = 1e30;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                auto it = grid.find(key(cx + dx, cy + dy));
                if (it == grid.end()) continue;
                for (int i : it->second) {
                    const double d = std::hypot(pts[i].x - q.x, pts[i].y - q.y);
                    if (d < bestD) { bestD = d; bestI = i; }
                }
            }
        dist = bestD;
        return bestI;
    };

    // Pitch from the strong crossings themselves: median nearest-neighbour distance.
    std::vector<double> nn;
    for (size_t i = 0; i < pts.size(); i += std::max<size_t>(1, pts.size() / 1500)) {
        double best = 1e30;
        for (size_t k = 0; k < pts.size(); ++k) {
            if (k == i) continue;
            const double d = std::hypot(pts[i].x - pts[k].x, pts[i].y - pts[k].y);
            if (d < best) best = d;
        }
        nn.push_back(best);
    }
    std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
    const double s = nn[nn.size() / 2];
    if (s < 0.6 * pitch || s > 1.5 * pitch)
        throw std::runtime_error("the grid crossings are not evenly spaced - could not identify the "
                                 "grid pitch (estimate " + std::to_string(pitch) + " px, nearest "
                                 "neighbours " + std::to_string(s) + " px)");

    struct Node { int i, j; cv::Point2d ex, ey; };
    std::unordered_map<int, Node> idx;
    int seed = 0; double bestD = 1e30;
    for (size_t k = 0; k < pts.size(); ++k) {
        const double d = std::hypot(pts[k].x - frame.width / 2.0, pts[k].y - frame.height / 2.0);
        if (d < bestD) { bestD = d; seed = int(k); }
    }
    idx[seed] = {0, 0, {s, 0.0}, {0.0, s}};
    std::vector<int> queue{seed};
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int k = queue[qi];
        const Node nd = idx[k];
        const cv::Point2d p(pts[k].x, pts[k].y);
        const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
        for (int d = 0; d < 4; ++d) {
            const cv::Point2d step = d < 2 ? nd.ex * di[d] : nd.ey * dj[d];
            double dist;
            const int n = nearest(p + step, dist);
            if (n < 0 || dist > 0.28 * s || idx.count(n)) continue;
            Node nn2{nd.i + di[d], nd.j + dj[d], nd.ex, nd.ey};
            const cv::Point2d moved(pts[n].x - p.x, pts[n].y - p.y);
            if (d < 2) nn2.ex = moved * di[d]; else nn2.ey = moved * dj[d];
            idx[n] = nn2;
            queue.push_back(n);
        }
    }

    Lattice lat;
    lat.pitch = s;
    for (const auto& kv : idx) {
        lat.ij.emplace_back(kv.second.i, kv.second.j);
        lat.xy.emplace_back(pts[kv.first].x, pts[kv.first].y);
    }
    return lat;
}

// ------------------------------------------------------------------- fitting

cv::Mat polyBasis(const std::vector<double>& u, const std::vector<double>& v, int degree)
{
    int terms = (degree + 1) * (degree + 2) / 2;
    cv::Mat B(int(u.size()), terms, CV_64F);
    for (int r = 0; r < B.rows; ++r) {
        int c = 0;
        for (int a = 0; a <= degree; ++a)
            for (int b = 0; a + b <= degree; ++b)
                B.at<double>(r, c++) = std::pow(u[r], a) * std::pow(v[r], b);
    }
    return B;
}

} // namespace

QString DistortionCorrection::Report::summary() const
{
    return QString("Distortion correction: %1 grid image(s), %2 crossings (%3 x %4), grid pitch %5 px, "
                   "grid rotation %6 deg (kept). Deviation from a square grid: RMS %7 px (max %8) -> "
                   "RMS %9 px (max %10) after correction")
        .arg(imageCount).arg(crossings).arg(gridColumns).arg(gridRows).arg(pitchPx, 0, 'f', 2)
        .arg(rotationDeg, 0, 'f', 2).arg(rmsBeforePx, 0, 'f', 2).arg(maxBeforePx, 0, 'f', 1)
        .arg(rmsAfterPx, 0, 'f', 2).arg(maxAfterPx, 0, 'f', 1);
}

std::vector<cv::Mat> DistortionCorrection::loadGridImages(const QString& dir, int rotateDeg, bool flipRows)
{
    QDir d(dir);
    if (!d.exists())
        throw std::runtime_error("grid folder not found: " + dir.toStdString());
    const QStringList files = d.entryList(QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
    std::vector<cv::Mat> images;
    for (const QString& name : files) {
        if (name.startsWith('.') || name.endsWith(".txt", Qt::CaseInsensitive))
            continue;
        const QString path = d.filePath(name);
        cv::Mat img;
        const QString ext = QFileInfo(name).suffix().toLower();
        try {
            if (ext == "fit" || ext == "fits" || ext == "fts")
                img = readFits(path);
            else
                img = toGrayFloat(cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("grid image ") + name.toStdString() + ": " + e.what());
        }
        if (img.empty())
            continue; // not an image (e.g. a stray text file)
        if (!images.empty() && img.size() != images.front().size())
            throw std::runtime_error("grid image " + name.toStdString() + " has a different size than the others");
        images.push_back(img);
    }
    if (images.empty())
        throw std::runtime_error("no readable images in " + dir.toStdString());

    for (cv::Mat& img : images) {
        switch (((rotateDeg % 360) + 360) % 360) {
        case 90: cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(img, img, cv::ROTATE_180); break;
        case 270: cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default: break;
        }
        if (flipRows)
            cv::flip(img, img, 0);
    }
    return images;
}

std::shared_ptr<const DistortionCorrection> DistortionCorrection::fit(const std::vector<cv::Mat>& images,
                                                                      const Options& options, Report* report)
{
    if (images.empty())
        throw std::runtime_error("no grid images");
    cv::Mat avg = cv::Mat::zeros(images[0].size(), CV_32FC1);
    for (const cv::Mat& im : images) {
        if (im.size() != images[0].size())
            throw std::runtime_error("the grid images differ in size");
        cv::Mat f = toGrayFloat(im);
        cv::accumulate(f, avg);
    }
    avg /= double(images.size());

    // Flatten the illumination (lines are dark on the paper).
    cv::Mat bg, flat;
    cv::GaussianBlur(avg, bg, cv::Size(0, 0), 30.0);
    cv::max(bg, 1.0f, bg);
    cv::divide(avg, bg, flat);
    bg.release();
    avg.release();

    double pitch = estimatePitch(flat);
    std::vector<Crossing> pts = detectCrossings(flat, pitch);
    Lattice lat = indexLattice(pts, flat.size(), pitch);
    if (std::abs(lat.pitch - pitch) > 0.2 * pitch) { // first guess was off (e.g. a harmonic): redo with the measured one
        pts = detectCrossings(flat, lat.pitch);
        lat = indexLattice(pts, flat.size(), lat.pitch);
    }
    flat.release();
    const size_t N = lat.ij.size();
    if (N < 100)
        throw std::runtime_error("only " + std::to_string(N) + " grid crossings could be matched to a "
                                 "regular grid - need a sharp image of graph paper covering much of the frame");

    std::shared_ptr<DistortionCorrection> model(new DistortionCorrection()); // private ctor
    DistortionCorrection& m = *model;
    m.frameSize_ = images[0].size();
    m.degree_ = std::max(1, std::min(options.polyDegree, 8));
    m.pitch_ = lat.pitch;

    // Similarity (rotation + uniform scale + shift) = the ideal square grid in the image:
    // w = A z + B with z = i + j*I, w = x + y*I (least squares over the complex numbers).
    std::vector<char> keep(N, 1);
    std::complex<double> A(1, 0), B(0, 0);
    auto fitSimilarity = [&]() {
        std::complex<double> zm = 0, wm = 0; double cnt = 0;
        for (size_t k = 0; k < N; ++k) if (keep[k]) {
            zm += std::complex<double>(lat.ij[k].x, lat.ij[k].y);
            wm += std::complex<double>(lat.xy[k].x, lat.xy[k].y); cnt++;
        }
        zm /= cnt; wm /= cnt;
        std::complex<double> num = 0; double den = 0;
        for (size_t k = 0; k < N; ++k) if (keep[k]) {
            const std::complex<double> dz = std::complex<double>(lat.ij[k].x, lat.ij[k].y) - zm;
            const std::complex<double> dw = std::complex<double>(lat.xy[k].x, lat.xy[k].y) - wm;
            num += std::conj(dz) * dw; den += std::norm(dz);
        }
        A = num / den; B = wm - A * zm;
    };

    std::vector<double> u(N), v(N);
    cv::Mat coef; // (3 affine + poly terms) x 2
    for (int round = 0; round < 5; ++round) {
        // Lattice centre/extent from the kept points.
        double si = 0, sj = 0, cnt = 0;
        m.iMin_ = m.jMin_ = 1e30; m.iMax_ = m.jMax_ = -1e30;
        for (size_t k = 0; k < N; ++k) if (keep[k]) {
            si += lat.ij[k].x; sj += lat.ij[k].y; cnt++;
            m.iMin_ = std::min(m.iMin_, lat.ij[k].x); m.iMax_ = std::max(m.iMax_, lat.ij[k].x);
            m.jMin_ = std::min(m.jMin_, lat.ij[k].y); m.jMax_ = std::max(m.jMax_, lat.ij[k].y);
        }
        m.i0_ = si / cnt; m.j0_ = sj / cnt;
        fitSimilarity();

        // Affine part first, then the polynomial on its residual (affine + poly together would be
        // degenerate in the low-order terms).
        cv::Mat Aff((int)cnt, 3, CV_64F), X((int)cnt, 2, CV_64F);
        std::vector<double> uk, vk;
        int r = 0;
        for (size_t k = 0; k < N; ++k) if (keep[k]) {
            Aff.at<double>(r, 0) = lat.ij[k].x; Aff.at<double>(r, 1) = lat.ij[k].y; Aff.at<double>(r, 2) = 1;
            X.at<double>(r, 0) = lat.xy[k].x; X.at<double>(r, 1) = lat.xy[k].y;
            uk.push_back((lat.ij[k].x - m.i0_) / m.pitch_); vk.push_back((lat.ij[k].y - m.j0_) / m.pitch_);
            r++;
        }
        cv::Mat affSol;
        cv::solve(Aff, X, affSol, cv::DECOMP_SVD);
        cv::Mat resid = X - Aff * affSol;
        cv::Mat B1 = polyBasis(uk, vk, m.degree_);
        cv::Mat polySol;
        cv::solve(B1, resid, polySol, cv::DECOMP_SVD);
        cv::Mat fitted = Aff * affSol + B1 * polySol;
        for (int c = 0; c < 3; ++c) { m.affine_(c, 0) = affSol.at<double>(c, 0); m.affine_(c, 1) = affSol.at<double>(c, 1); }
        m.poly_ = polySol.clone();

        // Outlier rejection (mis-indexed or smudged crossings).
        std::vector<double> err;
        for (int k = 0; k < (int)cnt; ++k)
            err.push_back(std::hypot(X.at<double>(k, 0) - fitted.at<double>(k, 0), X.at<double>(k, 1) - fitted.at<double>(k, 1)));
        std::vector<double> sorted = err;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const double mad = sorted[sorted.size() / 2];
        const double limit = std::max(6.0 * 1.4826 * mad, 3.0);
        int changed = 0, kk = 0;
        for (size_t k = 0; k < N; ++k) if (keep[k]) {
            if (err[kk] > limit) { keep[k] = 0; changed++; }
            kk++;
        }
        if (!changed) break;
    }

    m.simAr_ = A.real(); m.simAi_ = A.imag(); m.simBx_ = B.real(); m.simBy_ = B.imag();

    if (report) {
        report->imageCount = int(images.size());
        report->pitchPx = std::abs(A);
        report->rotationDeg = std::arg(A) * 180.0 / CV_PI;
        double s1 = 0, s2 = 0, mx1 = 0, mx2 = 0; int n = 0;
        for (size_t k = 0; k < N; ++k) if (keep[k]) {
            const std::complex<double> w = A * std::complex<double>(lat.ij[k].x, lat.ij[k].y) + B;
            const double e1 = std::hypot(w.real() - lat.xy[k].x, w.imag() - lat.xy[k].y);
            const double ui = (lat.ij[k].x - m.i0_) / m.pitch_, vj = (lat.ij[k].y - m.j0_) / m.pitch_;
            cv::Mat Bk = polyBasis({ui}, {vj}, m.degree_);
            cv::Mat pk = Bk * m.poly_;
            const double mxp = lat.ij[k].x * m.affine_(0, 0) + lat.ij[k].y * m.affine_(1, 0) + m.affine_(2, 0) + pk.at<double>(0, 0);
            const double myp = lat.ij[k].x * m.affine_(0, 1) + lat.ij[k].y * m.affine_(1, 1) + m.affine_(2, 1) + pk.at<double>(0, 1);
            const double e2 = std::hypot(mxp - lat.xy[k].x, myp - lat.xy[k].y);
            s1 += e1 * e1; s2 += e2 * e2; mx1 = std::max(mx1, e1); mx2 = std::max(mx2, e2); n++;
        }
        report->crossings = n;
        report->gridColumns = int(m.iMax_ - m.iMin_) + 1;
        report->gridRows = int(m.jMax_ - m.jMin_) + 1;
        report->rmsBeforePx = std::sqrt(s1 / n); report->maxBeforePx = mx1;
        report->rmsAfterPx = std::sqrt(s2 / n); report->maxAfterPx = mx2;
    }
    return model;
}

cv::Point2d DistortionCorrection::sourcePosition(double x, double y) const
{
    // Corrected-image pixel -> lattice coordinates through the inverse of the ideal-grid similarity.
    const std::complex<double> A(simAr_, simAi_), B(simBx_, simBy_);
    const std::complex<double> z = (std::complex<double>(x, y) - B) / A;
    const double i = z.real(), j = z.imag();

    // Polynomial part held at its edge value outside the measured lattice; the affine part simply
    // continues.
    const double ic = std::min(std::max(i, iMin_), iMax_), jc = std::min(std::max(j, jMin_), jMax_);
    const double u = (ic - i0_) / pitch_, v = (jc - j0_) / pitch_;
    double up[9], vp[9];
    up[0] = vp[0] = 1.0;
    for (int a = 1; a <= degree_; ++a) { up[a] = up[a - 1] * u; vp[a] = vp[a - 1] * v; }
    double px = 0, py = 0;
    int c = 0;
    for (int a = 0; a <= degree_; ++a)
        for (int b = 0; a + b <= degree_; ++b, ++c) {
            const double t = up[a] * vp[b];
            px += t * poly_.at<double>(c, 0);
            py += t * poly_.at<double>(c, 1);
        }
    return {i * affine_(0, 0) + j * affine_(1, 0) + affine_(2, 0) + px,
            i * affine_(0, 1) + j * affine_(1, 1) + affine_(2, 1) + py};
}

cv::Rect DistortionCorrection::requiredSourceRect(const cv::Rect& roiIn, int binning) const
{
    const int b = std::max(1, binning);
    cv::Rect roi(roiIn.x, roiIn.y, (roiIn.width / b) * b, (roiIn.height / b) * b);
    double minX = 1e30, minY = 1e30, maxX = -1e30, maxY = -1e30;
    const int steps = 48;
    for (int a = 0; a <= steps; ++a)
        for (int c = 0; c <= steps; ++c) {
            const cv::Point2d p = sourcePosition(roi.x + roi.width * double(a) / steps, roi.y + roi.height * double(c) / steps);
            minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
        }
    const double margin = 4.0; // bilinear footprint plus slack for what the sampling above can miss
    auto grow = [&](int start, int len, double lo, double hi, int frame) {
        int before = int(std::ceil((start - (lo - margin)) / b)); // blocks to add on the left
        int after = int(std::ceil(((hi + margin) - (start + len)) / b));
        before = std::max(0, std::min(before, start / b));
        after = std::max(0, std::min(after, (frame - (start + len)) / b));
        return std::make_pair(start - before * b, len + (before + after) * b);
    };
    const auto xs = grow(roi.x, roi.width, minX, maxX, frameSize_.width);
    const auto ys = grow(roi.y, roi.height, minY, maxY, frameSize_.height);
    return cv::Rect(xs.first, ys.first, xs.second, ys.second);
}

cv::Mat DistortionCorrection::buildRemap(const cv::Rect& roiIn, const cv::Rect& srcRect, int binning) const
{
    const int b = std::max(1, binning);
    const int w = roiIn.width / b, h = roiIn.height / b;
    cv::Mat map(h, w, CV_32FC2);
    for (int yb = 0; yb < h; ++yb) {
        cv::Vec2f* row = map.ptr<cv::Vec2f>(yb);
        for (int xb = 0; xb < w; ++xb) {
            // Centre of the binned output pixel in full-resolution coordinates, ...
            const double X = roiIn.x + (xb + 0.5) * b - 0.5, Y = roiIn.y + (yb + 0.5) * b - 0.5;
            const cv::Point2d s = sourcePosition(X, Y);
            // ... and where that source position lies in the binned image of srcRect.
            row[xb] = cv::Vec2f(float((s.x - srcRect.x + 0.5) / b - 0.5), float((s.y - srcRect.y + 0.5) / b - 0.5));
        }
    }
    return map;
}
