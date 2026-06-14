#include "CushionDetection.h"
#include <iostream>
#include <cmath>

// Table specifications (standard playing surface dimensions)
static constexpr TableSpec kTableSpecs[] = {
    { TableSize::SevenFt,     "7ft (Bar)",        39.0f, 78.0f },
    { TableSize::EightFt,     "8ft (Home)",       44.0f, 88.0f },
    { TableSize::EightHalfFt, "8.5ft (Pro-Am)",   46.0f, 92.0f },
    { TableSize::NineFt,      "9ft (Tournament)", 50.0f, 100.0f }
};

// Cushion specifications (nose line inset defaults)
static constexpr CushionSpec kCushionSpecs[] = {
    { CushionType::K66,     "Brunswick K-66",      1.50f },
    { CushionType::K55,     "Brunswick K-55",      1.50f },
    { CushionType::U23,     "Artemis U-23",        1.50f },
    { CushionType::U56,     "Super Aramith U-56",  1.50f },
    { CushionType::Default, "Default",             1.50f }
};

// Get table specification
TableSpec getTableSpec(TableSize size) {
    for (const auto& spec : kTableSpecs) {
        if (spec.size == size) {
            return spec;
        }
    }
    // Default to 9ft if not found
    return kTableSpecs[3];
}

// Get cushion specification
CushionSpec getCushionSpec(CushionType type) {
    for (const auto& spec : kCushionSpecs) {
        if (spec.type == type) {
            return spec;
        }
    }
    // Default to K66 if not found
    return kCushionSpecs[0];
}

// Helper: Compute line equation ax + by + c = 0 from two points
struct Line {
    float a, b, c;
};

static Line computeLine(const cv::Point2f& p1, const cv::Point2f& p2) {
    Line line;
    // Direction vector
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;

    // Normal vector (perpendicular)
    line.a = -dy;
    line.b = dx;

    // Normalize to unit normal
    float norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm > 1e-6f) {
        line.a /= norm;
        line.b /= norm;
    }

    // Compute c from point
    line.c = -(line.a * p1.x + line.b * p1.y);

    return line;
}

// Helper: Signed distance from point to line
static float signedDistance(const cv::Point2f& p, const Line& line) {
    return line.a * p.x + line.b * p.y + line.c;
}

// Helper: Offset line by distance (positive = in direction of normal)
static Line offsetLine(const Line& line, float dist) {
    Line offsetted = line;
    offsetted.c -= dist;  // Move in direction of normal
    return offsetted;
}

// Helper: Intersect two lines
static cv::Point2f intersectLines(const Line& line1, const Line& line2) {
    // Solve: a1*x + b1*y + c1 = 0
    //        a2*x + b2*y + c2 = 0
    float det = line1.a * line2.b - line2.a * line1.b;

    if (std::abs(det) < 1e-6f) {
        // Lines are parallel - return midpoint of c values
        return cv::Point2f(0, 0);
    }

    float x = (line1.b * line2.c - line2.b * line1.c) / det;
    float y = (line2.a * line1.c - line1.a * line2.c) / det;

    return cv::Point2f(x, y);
}

// Compute playable area in rectified space
PlayableAreaResult computePlayableAreaRectified(
    const cv::Size& rectifiedSize,
    const std::array<cv::Point2f, 4>& feltCornersRectified,
    TableSize tableSize,
    CushionType cushionType,
    bool overrideInset,
    float overrideInsetIn
) {
    PlayableAreaResult result;

    // Get specifications
    TableSpec tableSpec = getTableSpec(tableSize);
    CushionSpec cushionSpec = getCushionSpec(cushionType);

    // Extract corners (assumed order: TL, TR, BR, BL)
    const cv::Point2f& TL = feltCornersRectified[0];
    const cv::Point2f& TR = feltCornersRectified[1];
    const cv::Point2f& BR = feltCornersRectified[2];
    const cv::Point2f& BL = feltCornersRectified[3];

    // Compute felt dimensions in pixels
    float topEdgePx = cv::norm(TR - TL);
    float bottomEdgePx = cv::norm(BR - BL);
    float leftEdgePx = cv::norm(BL - TL);
    float rightEdgePx = cv::norm(BR - TR);

    float rectifiedFeltWpx = (topEdgePx + bottomEdgePx) * 0.5f;
    float rectifiedFeltHpx = (leftEdgePx + rightEdgePx) * 0.5f;

    std::cout << "[Playable Area] Felt dimensions: " << rectifiedFeltWpx << " x " << rectifiedFeltHpx << " px\n";

    // Compute px-per-inch scaling
    result.pxPerInX = rectifiedFeltWpx / tableSpec.playWidthIn;
    result.pxPerInY = rectifiedFeltHpx / tableSpec.playHeightIn;
    result.pxPerIn = (result.pxPerInX + result.pxPerInY) * 0.5f;

    std::cout << "[Playable Area] Scaling: " << result.pxPerInX << " px/in (X), "
              << result.pxPerInY << " px/in (Y), " << result.pxPerIn << " px/in (avg)\n";

    // Determine nose inset
    float insetIn = overrideInset ? overrideInsetIn : cushionSpec.noseInsetInDefault;
    result.insetPx = insetIn * result.pxPerIn;

    std::cout << "[Playable Area] Nose inset: " << insetIn << " in = " << result.insetPx << " px\n";

    // Compute quad center (for determining inward direction)
    cv::Point2f center = (TL + TR + BR + BL) * 0.25f;

    // Compute edge lines
    Line topLine = computeLine(TL, TR);
    Line rightLine = computeLine(TR, BR);
    Line bottomLine = computeLine(BR, BL);
    Line leftLine = computeLine(BL, TL);

    // Determine inward offset direction for each edge
    // We want to offset toward the center
    float topDist = signedDistance(center, topLine);
    float rightDist = signedDistance(center, rightLine);
    float bottomDist = signedDistance(center, bottomLine);
    float leftDist = signedDistance(center, leftLine);

    // Offset lines inward (toward center)
    Line topLineOff = offsetLine(topLine, topDist > 0 ? result.insetPx : -result.insetPx);
    Line rightLineOff = offsetLine(rightLine, rightDist > 0 ? result.insetPx : -result.insetPx);
    Line bottomLineOff = offsetLine(bottomLine, bottomDist > 0 ? result.insetPx : -result.insetPx);
    Line leftLineOff = offsetLine(leftLine, leftDist > 0 ? result.insetPx : -result.insetPx);

    // Compute inset corners by intersecting adjacent offset lines
    result.noseCornersRect[0] = intersectLines(topLineOff, leftLineOff);    // TL
    result.noseCornersRect[1] = intersectLines(topLineOff, rightLineOff);   // TR
    result.noseCornersRect[2] = intersectLines(bottomLineOff, rightLineOff); // BR
    result.noseCornersRect[3] = intersectLines(bottomLineOff, leftLineOff);  // BL

    // Validate inset corners (should be inside original quad)
    bool valid = true;
    for (const auto& corner : result.noseCornersRect) {
        if (corner.x < 0 || corner.x >= rectifiedSize.width ||
            corner.y < 0 || corner.y >= rectifiedSize.height) {
            valid = false;
            std::cerr << "[Playable Area] ERROR: Inset corner out of bounds: " << corner << "\n";
        }
    }

    if (!valid) {
        std::cerr << "[Playable Area] ERROR: Invalid inset corners\n";
        return result;
    }

    // Compute playable dimensions
    float playableTopPx = cv::norm(result.noseCornersRect[1] - result.noseCornersRect[0]);
    float playableBottomPx = cv::norm(result.noseCornersRect[2] - result.noseCornersRect[3]);
    float playableLeftPx = cv::norm(result.noseCornersRect[3] - result.noseCornersRect[0]);
    float playableRightPx = cv::norm(result.noseCornersRect[2] - result.noseCornersRect[1]);

    result.playableWidthPx = (playableTopPx + playableBottomPx) * 0.5f;
    result.playableHeightPx = (playableLeftPx + playableRightPx) * 0.5f;

    std::cout << "[Playable Area] Playable dimensions: " << result.playableWidthPx << " x "
              << result.playableHeightPx << " px\n";

    // Build playable mask
    result.playableMask = cv::Mat::zeros(rectifiedSize, CV_8U);

    std::vector<cv::Point> polyPoints;
    for (const auto& pt : result.noseCornersRect) {
        polyPoints.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    cv::fillConvexPoly(result.playableMask, polyPoints.data(), static_cast<int>(polyPoints.size()), cv::Scalar(255));

    // Optional: Erode by 1px for safety
    cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(result.playableMask, result.playableMask, erodeKernel);

    result.ok = true;
    std::cout << "[Playable Area] Computation succeeded\n";

    return result;
}

// Draw playable area overlay
void drawPlayableAreaOverlayRectified(
    cv::Mat& rectifiedBgr,
    const PlayableAreaResult& result,
    const cv::Scalar& color,
    int thickness,
    bool filled,
    int alpha
) {
    if (!result.ok || rectifiedBgr.empty()) {
        return;
    }

    // Convert corners to integer points
    std::vector<cv::Point> polyPoints;
    for (const auto& pt : result.noseCornersRect) {
        polyPoints.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    if (filled && alpha > 0) {
        // Draw filled polygon with transparency
        cv::Mat overlay = rectifiedBgr.clone();
        cv::fillConvexPoly(overlay, polyPoints.data(), static_cast<int>(polyPoints.size()), color);

        float blendAlpha = alpha / 255.0f;
        cv::addWeighted(overlay, blendAlpha, rectifiedBgr, 1.0f - blendAlpha, 0, rectifiedBgr);
    }

    // Always draw outline
    cv::polylines(rectifiedBgr, polyPoints, true, color, thickness, cv::LINE_AA);

    // Draw corner markers
    for (const auto& pt : result.noseCornersRect) {
        cv::circle(rectifiedBgr, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                  5, color, -1, cv::LINE_AA);
    }
}

// Project playable area to perspective view
PlayableAreaResult projectPlayableAreaToPerspective(
    const PlayableAreaResult& rectifiedResult,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
) {
    PlayableAreaResult perspResult;

    if (!rectifiedResult.ok || Hinv.empty()) {
        return perspResult;
    }

    // Copy scalar values
    perspResult.pxPerInX = rectifiedResult.pxPerInX;
    perspResult.pxPerInY = rectifiedResult.pxPerInY;
    perspResult.pxPerIn = rectifiedResult.pxPerIn;
    perspResult.insetPx = rectifiedResult.insetPx;
    perspResult.playableWidthPx = rectifiedResult.playableWidthPx;
    perspResult.playableHeightPx = rectifiedResult.playableHeightPx;

    // Project corners from rectified to perspective
    std::vector<cv::Point2f> rectifiedCorners(rectifiedResult.noseCornersRect.begin(),
                                               rectifiedResult.noseCornersRect.end());
    std::vector<cv::Point2f> perspectiveCorners;
    cv::perspectiveTransform(rectifiedCorners, perspectiveCorners, Hinv);

    std::copy(perspectiveCorners.begin(), perspectiveCorners.end(), perspResult.noseCornersRect.begin());

    // Build mask in perspective space
    perspResult.playableMask = cv::Mat::zeros(perspectiveSize, CV_8U);

    std::vector<cv::Point> polyPoints;
    for (const auto& pt : perspResult.noseCornersRect) {
        polyPoints.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    cv::fillConvexPoly(perspResult.playableMask, polyPoints.data(),
                      static_cast<int>(polyPoints.size()), cv::Scalar(255));

    perspResult.ok = true;

    return perspResult;
}
