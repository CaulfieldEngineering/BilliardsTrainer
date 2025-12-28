#pragma once
#include <opencv2/core.hpp>
#include <array>

struct RectificationResult {
    bool ok = false;
    cv::Mat rectifiedBgr;
    cv::Mat rectifiedFeltMask;
    cv::Mat H;     // 3x3 CV_64F
    cv::Mat Hinv;  // 3x3 CV_64F
    cv::Size dstSize;
    
    // Debug information
    std::array<cv::Point2f, 4> srcQuad;        // Original felt corners
    std::array<cv::Point2f, 4> expandedSrcQuad; // Expanded source quad
    std::array<cv::Point2f, 4> dstQuad;        // Destination quad (with padding)
};

RectificationResult rectifyTabletop(
    const cv::Mat& bgr,
    const cv::Mat& feltMask,
    const std::array<cv::Point2f,4>& cornersTLTRBRBL,
    float rectifyMarginScale = 1.18f,
    int rectifyPadPx = 40
);

