#pragma once

#include <opencv2/opencv.hpp>
#include <array>

// Table sizes (standard pool table sizes)
enum class TableSize {
    SevenFt,      // 7-foot (bar table)
    EightFt,      // 8-foot (home table)
    EightHalfFt,  // 8.5-foot (pro-am)
    NineFt        // 9-foot (tournament)
};

// Cushion types (common profiles)
enum class CushionType {
    K66,      // Brunswick K-66 (most common tournament cushion)
    K55,      // Brunswick K-55 (older style)
    U23,      // Artemis U-23 (European style)
    U56,      // Super Aramith U-56 (modern European)
    Default   // Generic/unknown cushion
};

// Table specification (playing surface dimensions)
struct TableSpec {
    TableSize size;
    const char* label;
    float playWidthIn;   // Playing surface width (inches)
    float playHeightIn;  // Playing surface height (inches)
};

// Cushion specification (nose line inset)
struct CushionSpec {
    CushionType type;
    const char* label;
    float noseInsetInDefault;  // Distance from cloth edge to nose line (inches)
    // Future: Add more detailed cushion geometry if needed
    // float noseHeightIn;
    // float baseWidthIn;
    // float facingAngleDeg;
};

// Result structure for playable area computation
struct PlayableAreaResult {
    bool ok = false;                              // True if computation succeeded

    // Scaling factors (px per inch in rectified space)
    float pxPerInX = 0.0f;                        // Horizontal px/in
    float pxPerInY = 0.0f;                        // Vertical px/in
    float pxPerIn = 0.0f;                         // Average px/in

    // Inset amount
    float insetPx = 0.0f;                         // Nose line inset in pixels

    // Playable area geometry (rectified space)
    std::array<cv::Point2f, 4> noseCornersRect;  // Inset corners: TL, TR, BR, BL
    cv::Mat playableMask;                         // CV_8U (0/255), same size as rectified image

    // Computed dimensions
    float playableWidthPx = 0.0f;
    float playableHeightPx = 0.0f;
};

// Get table specification for a given size
TableSpec getTableSpec(TableSize size);

// Get cushion specification for a given type
CushionSpec getCushionSpec(CushionType type);

// Compute playable area (nose line boundary) in rectified space
// Inputs:
//   - rectifiedSize: Size of the rectified image
//   - feltCornersRectified: Felt boundary corners in rectified space (TL, TR, BR, BL)
//   - tableSize: Selected table size
//   - cushionType: Selected cushion type
//   - overrideInset: If true, use overrideInsetIn instead of cushion default
//   - overrideInsetIn: Custom nose line inset (inches)
PlayableAreaResult computePlayableAreaRectified(
    const cv::Size& rectifiedSize,
    const std::array<cv::Point2f, 4>& feltCornersRectified,
    TableSize tableSize,
    CushionType cushionType,
    bool overrideInset,
    float overrideInsetIn
);

// Draw playable area overlay on rectified image
void drawPlayableAreaOverlayRectified(
    cv::Mat& rectifiedBgr,
    const PlayableAreaResult& result,
    const cv::Scalar& color,
    int thickness,
    bool filled,
    int alpha  // 0-255
);

// Project playable area from rectified to perspective view
PlayableAreaResult projectPlayableAreaToPerspective(
    const PlayableAreaResult& rectifiedResult,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
);
