#pragma once

// Orientation detection / rendering helpers extracted from main.cpp.
//
// This module owns the "Orientation Mask" computation (outer/inner quads, rail regions,
// and long/short rail identification by middle-pocket presence) and the rail-mask based
// dominant-axis estimator (TableOrientation).
//
// IMPORTANT:
// - No Win32 UI code should live here. The UI toggle/checkbox remains in main.cpp.
// - This file is OpenCV-only and can be used by any frontend.

#include <opencv2/core.hpp>

#include <vector>

// Two dominant axes (in image space) estimated from the rail mask.
struct TableOrientation {
    bool valid = false;
    double thetaA = 0.0;   // First axis angle in [0, pi)
    double thetaB = 0.0;   // Second axis angle in [0, pi)
    cv::Point2f dirA{};    // Unit direction vector for axis A
    cv::Point2f dirB{};    // Unit direction vector for axis B
};

// Parameters controlling how the Orientation Mask overlay is rendered.
struct OrientationMaskRenderParams {
    // Quads and divider lines
    cv::Scalar lineColorBGR = cv::Scalar(255, 0, 255); // purple
    int lineThicknessPx = 5;

    // Region fill
    cv::Scalar fillColorBGR = cv::Scalar(255, 0, 255); // purple
    double fillAlpha = 0.30; // 0..1

    // Inner quad margin: after inner quad is computed, shrink it toward its centroid
    // by this fraction (e.g. 0.05 == 5%).
    float innerShrinkFraction = 0.05f;

    // Text (L1/L2/S1/S2)
    double labelFontScale = 1.5;
    int labelThicknessPx = 3;
};

// Compute table orientation (two dominant axes) from the rail mask.
// This does NOT render anything.
TableOrientation computeTableOrientationFromRailMask(const cv::Mat& railMask);

// Draw the "Orientation Mask" overlay onto an existing BGR frame.
//
// Inputs:
// - frameBGR: destination frame to draw on (modified in place)
// - railMask: binary mask of rails (non-zero = rail pixels)
// - feltContour: detected felt boundary contour (used to compute inner quad)
// - params: render tuning knobs (colors, thickness, alpha, etc.)
//
// This function:
// - builds an outer quad from the rail mask boundary
// - builds an inner quad hugging the felt boundary and shrinks it slightly inward
// - partitions the donut into 4 rail regions using corner-to-corner dividers
// - identifies long rails by detecting the mid-side pocket notch inside each region
// - shades regions and labels them L1/L2 (long) and S1/S2 (short)
void drawOrientationMaskOverlay(
    cv::Mat& frameBGR,
    const cv::Mat& railMask,
    const std::vector<cv::Point>& feltContour,
    const OrientationMaskRenderParams& params);


