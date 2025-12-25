#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// We accept these parameter structs so the sidebar tuning actually impacts diamond detection.
#include "felt_detection.h"
#include "rail_detection.h"

// Debug outputs to help diagnose "no diamonds" / "garbage diamonds" quickly.
// Filled by `detectDiamonds()` when a non-null debug pointer is supplied.
struct DiamondDebugImages {
    cv::Mat railMaskDark;    // output of detectRailMask(...)
    cv::Mat railSearchMask;  // filled/dilated version used as ROI
    cv::Mat railEnhanced;    // top-hat enhanced ROI (full-size for convenience)
    cv::Mat otsuBinary;      // fallback binary (full-size for convenience)
    cv::Mat diamondMask;     // final per-pixel mask for detected diamond/marker regions (full-size)
    cv::Rect roi;            // bounding rect of railSearchMask
    int keypointsFound = 0;  // SimpleBlobDetector keypoints count (pre-filter)
    int centersKept = 0;     // centers kept after mask + pocket rejection (pre-layout)
};

// Structure to hold detection parameters
struct DiamondDetectionParams {
    int threshold1 = 0;
    int threshold2 = 255;
    int minArea = 5;
    int maxArea = 1000;
    // Overlay styling
    cv::Scalar color = cv::Scalar(0, 255, 255); // BGR
    // Alpha (transparency) for rendering: 0..255 (0 = invisible, 255 = fully opaque)
    int alpha = 255;
    bool isFilled = true;
    int radiusPx = 8;
    int outlineThicknessPx = 2;
};

// Detect diamond markers on the pool table rails with context-aware constraints
void detectDiamonds(
    const cv::Mat& src,
    cv::Mat& dst,
    bool showDiamonds,
    const DiamondDetectionParams& diamondParams,
    const FeltParams& feltParams,
    const RailParams& railParams,
    DiamondDebugImages* debugOut = nullptr);
