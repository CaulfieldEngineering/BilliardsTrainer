#include "rail_detection.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

// Helper function: Extract a band region around the felt mask for a specific edge
// direction: 0=top, 1=right, 2=bottom, 3=left
static cv::Mat extractRailBand(
    const cv::Mat& rectifiedBgr,
    const cv::Mat& rectifiedFeltMask,
    int direction,
    int bandWidth
) {
    if (rectifiedBgr.empty() || rectifiedFeltMask.empty()) {
        return cv::Mat();
    }
    
    cv::Size imgSize = rectifiedBgr.size();
    cv::Mat bandMask = cv::Mat::zeros(imgSize, CV_8UC1);
    
    // Get the bounding rectangle of the felt mask
    cv::Rect feltBounds = cv::boundingRect(rectifiedFeltMask);
    
    // Define band region based on direction
    cv::Rect bandRect;
    switch (direction) {
        case 0: // Top rail - band above the felt
            {
                int topY = std::max(0, feltBounds.y - bandWidth);
                int topHeight = feltBounds.y - topY;  // Actual height after clamping
                if (topHeight > 0) {
                    bandRect = cv::Rect(0, topY, imgSize.width, topHeight);
                } else {
                    return cv::Mat();  // No space above felt
                }
            }
            break;
        case 1: // Right rail - band to the right of the felt
            {
                int rightX = feltBounds.x + feltBounds.width;
                int rightWidth = std::min(bandWidth, imgSize.width - rightX);
                if (rightWidth > 0 && rightX < imgSize.width) {
                    bandRect = cv::Rect(rightX, feltBounds.y, rightWidth, feltBounds.height);
                } else {
                    return cv::Mat();  // No space to the right of felt
                }
            }
            break;
        case 2: // Bottom rail - band below the felt
            {
                int bottomY = feltBounds.y + feltBounds.height;
                int bottomHeight = std::min(bandWidth, imgSize.height - bottomY);
                if (bottomHeight > 0 && bottomY < imgSize.height) {
                    bandRect = cv::Rect(0, bottomY, imgSize.width, bottomHeight);
                } else {
                    return cv::Mat();  // No space below felt
                }
            }
            break;
        case 3: // Left rail - band to the left of the felt
            {
                int leftX = std::max(0, feltBounds.x - bandWidth);
                int leftWidth = feltBounds.x - leftX;  // Actual width after clamping
                if (leftWidth > 0) {
                    bandRect = cv::Rect(leftX, feltBounds.y, leftWidth, feltBounds.height);
                } else {
                    return cv::Mat();  // No space to the left of felt
                }
            }
            break;
        default:
            return cv::Mat();
    }
    
    // Final clamp to ensure within image bounds (should already be, but be safe)
    bandRect &= cv::Rect(0, 0, imgSize.width, imgSize.height);
    
    // Create mask for the band region
    bandMask(bandRect).setTo(255);
    
    // Exclude felt mask region from band (rails are outside the felt)
    cv::Mat feltInverted;
    cv::bitwise_not(rectifiedFeltMask, feltInverted);
    cv::bitwise_and(bandMask, feltInverted, bandMask);
    
    return bandMask;
}

// Helper function: Detect rail in a specific band region
static cv::Mat detectRailInBand(
    const cv::Mat& rectifiedBgr,
    const cv::Mat& bandMask,
    const RailParams& params,
    int direction
) {
    if (rectifiedBgr.empty() || bandMask.empty()) {
        return cv::Mat();
    }

    cv::Mat railMask;

    // Use color-based segmentation if enabled and color has been picked
    if (params.useColorFilter && params.hasPickedColor) {
        // Extract the band region from BGR image
        cv::Mat bandBgr;
        rectifiedBgr.copyTo(bandBgr, bandMask);

        // Convert to HSV for color filtering
        cv::Mat bandHsv;
        cv::cvtColor(bandBgr, bandHsv, cv::COLOR_BGR2HSV);

        // Apply color range filter
        cv::Mat colorMask;

        // Handle hue wrapping (when hMin > hMax, the range wraps around 180)
        if (params.colorHMin > params.colorHMax) {
            // Wrapped range: [hMin, 180] OR [0, hMax]
            cv::Mat mask1, mask2;
            cv::inRange(bandHsv,
                       cv::Scalar(params.colorHMin, params.colorSMin, params.colorVMin),
                       cv::Scalar(180, params.colorSMax, params.colorVMax),
                       mask1);
            cv::inRange(bandHsv,
                       cv::Scalar(0, params.colorSMin, params.colorVMin),
                       cv::Scalar(params.colorHMax, params.colorSMax, params.colorVMax),
                       mask2);
            cv::bitwise_or(mask1, mask2, colorMask);
        } else {
            // Normal range
            cv::inRange(bandHsv,
                       cv::Scalar(params.colorHMin, params.colorSMin, params.colorVMin),
                       cv::Scalar(params.colorHMax, params.colorSMax, params.colorVMax),
                       colorMask);
        }

        // Apply morphological operations to clean up and fill gaps
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(colorMask, colorMask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(colorMask, colorMask, cv::MORPH_OPEN, kernel);

        // Mask to band region
        cv::bitwise_and(colorMask, bandMask, railMask);
    } else {
        // Fallback: use the entire band region as the rail
        // This is simple but reliable - the band is already constrained to the rail area
        railMask = bandMask.clone();
    }

    // Clean up small noise using connected components
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(railMask, labels, stats, centroids, 8, CV_32S);

    if (nLabels > 1) {
        cv::Mat filteredMask = cv::Mat::zeros(railMask.size(), CV_8UC1);
        for (int i = 1; i < nLabels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area >= params.railMinArea) {
                cv::Mat componentMask = (labels == i);
                filteredMask.setTo(255, componentMask);
            }
        }
        railMask = filteredMask;
    }

    return railMask;
}

// Helper function: Project a single mask from rectified to perspective view
static cv::Mat projectMaskToPerspective(
    const cv::Mat& rectifiedMask,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
) {
    if (rectifiedMask.empty() || Hinv.empty() || Hinv.rows != 3 || Hinv.cols != 3) {
        return cv::Mat();
    }
    
    cv::Mat perspectiveMask;
    cv::warpPerspective(
        rectifiedMask,
        perspectiveMask,
        Hinv,
        perspectiveSize,
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    
    return perspectiveMask;
}

// Helper function: Project a contour from rectified to perspective view
static std::vector<cv::Point> projectContourToPerspective(
    const std::vector<cv::Point>& rectifiedContour,
    const cv::Mat& Hinv
) {
    if (rectifiedContour.empty() || Hinv.empty() || Hinv.rows != 3 || Hinv.cols != 3) {
        return std::vector<cv::Point>();
    }
    
    // Convert points to Point2f for perspective transform
    std::vector<cv::Point2f> rectifiedPoints2f;
    for (const auto& pt : rectifiedContour) {
        rectifiedPoints2f.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
    }
    
    // Apply perspective transformation
    std::vector<cv::Point2f> perspectivePoints2f;
    cv::perspectiveTransform(rectifiedPoints2f, perspectivePoints2f, Hinv);
    
    // Convert back to Point
    std::vector<cv::Point> perspectiveContour;
    for (const auto& pt : perspectivePoints2f) {
        perspectiveContour.push_back(cv::Point(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y))));
    }
    
    return perspectiveContour;
}

// Main rail detection function
RailDetectionResult detectRails(
    const cv::Mat& rectifiedBgr,
    const cv::Mat& rectifiedFeltMask,
    const RailParams& params
) {
    RailDetectionResult result;
    result.ok = false;
    result.numRailsDetected = 0;

    // Validate inputs
    if (rectifiedBgr.empty() || rectifiedFeltMask.empty()) {
        return result;
    }

    if (rectifiedBgr.size() != rectifiedFeltMask.size()) {
        return result;
    }

    // Initialize result structures
    result.railMask = cv::Mat::zeros(rectifiedBgr.size(), CV_8UC1);
    result.railMasks.resize(4);
    result.railOutlines.resize(4);
    result.railBoundingBoxes.resize(4);

    // Get felt bounding box - this defines the inner boundary of the rails
    cv::Rect feltBounds = cv::boundingRect(rectifiedFeltMask);

    // Step 1: Measure the actual rail width using multiple scanlines on all 4 sides
    // Use argmax gradient for robust edge detection
    int measuredRailWidth = params.railMaxWidth; // Default fallback

    {
        // Precompute grayscale once for efficiency
        cv::Mat gray;
        cv::cvtColor(rectifiedBgr, gray, cv::COLOR_BGR2GRAY);

        const int numSamples = 11;
        const int gradThresh = 30;

        std::vector<int> topWidths, bottomWidths, leftWidths, rightWidths;

        // Sample top edge (scan upward: y = feltTop - d)
        for (int i = 0; i < numSamples; i++) {
            int x = feltBounds.x + (feltBounds.width * i) / (numSamples - 1);
            int startY = feltBounds.y;

            int bestD = -1;
            int bestGrad = 0;

            for (int d = 0; d < params.railBandWidth; d++) {
                int y = startY - d;
                if (y < 1 || y >= gray.rows - 1) break;

                // Compute max vertical gradient along this horizontal line
                int maxGrad = 0;
                for (int sx = std::max(1, x - 2); sx <= std::min(gray.cols - 2, x + 2); sx++) {
                    int grad = std::abs(gray.at<uchar>(y - 1, sx) - gray.at<uchar>(y + 1, sx));
                    maxGrad = std::max(maxGrad, grad);
                }

                if (maxGrad > bestGrad) {
                    bestGrad = maxGrad;
                    bestD = d;
                }
            }

            if (bestGrad >= gradThresh && bestD > 0) {
                topWidths.push_back(bestD);
            }
        }

        // Sample bottom edge (scan downward: y = feltBottom + d)
        for (int i = 0; i < numSamples; i++) {
            int x = feltBounds.x + (feltBounds.width * i) / (numSamples - 1);
            int startY = feltBounds.y + feltBounds.height;

            int bestD = -1;
            int bestGrad = 0;

            for (int d = 0; d < params.railBandWidth; d++) {
                int y = startY + d;
                if (y < 1 || y >= gray.rows - 1) break;

                int maxGrad = 0;
                for (int sx = std::max(1, x - 2); sx <= std::min(gray.cols - 2, x + 2); sx++) {
                    int grad = std::abs(gray.at<uchar>(y - 1, sx) - gray.at<uchar>(y + 1, sx));
                    maxGrad = std::max(maxGrad, grad);
                }

                if (maxGrad > bestGrad) {
                    bestGrad = maxGrad;
                    bestD = d;
                }
            }

            if (bestGrad >= gradThresh && bestD > 0) {
                bottomWidths.push_back(bestD);
            }
        }

        // Sample left edge (scan leftward: x = feltLeft - d)
        for (int i = 0; i < numSamples; i++) {
            int y = feltBounds.y + (feltBounds.height * i) / (numSamples - 1);
            int startX = feltBounds.x;

            int bestD = -1;
            int bestGrad = 0;

            for (int d = 0; d < params.railBandWidth; d++) {
                int x = startX - d;
                if (x < 1 || x >= gray.cols - 1) break;

                int maxGrad = 0;
                for (int sy = std::max(1, y - 2); sy <= std::min(gray.rows - 2, y + 2); sy++) {
                    int grad = std::abs(gray.at<uchar>(sy, x - 1) - gray.at<uchar>(sy, x + 1));
                    maxGrad = std::max(maxGrad, grad);
                }

                if (maxGrad > bestGrad) {
                    bestGrad = maxGrad;
                    bestD = d;
                }
            }

            if (bestGrad >= gradThresh && bestD > 0) {
                leftWidths.push_back(bestD);
            }
        }

        // Sample right edge (scan rightward: x = feltRight + d)
        for (int i = 0; i < numSamples; i++) {
            int y = feltBounds.y + (feltBounds.height * i) / (numSamples - 1);
            int startX = feltBounds.x + feltBounds.width;

            int bestD = -1;
            int bestGrad = 0;

            for (int d = 0; d < params.railBandWidth; d++) {
                int x = startX + d;
                if (x < 1 || x >= gray.cols - 1) break;

                int maxGrad = 0;
                for (int sy = std::max(1, y - 2); sy <= std::min(gray.rows - 2, y + 2); sy++) {
                    int grad = std::abs(gray.at<uchar>(sy, x - 1) - gray.at<uchar>(sy, x + 1));
                    maxGrad = std::max(maxGrad, grad);
                }

                if (maxGrad > bestGrad) {
                    bestGrad = maxGrad;
                    bestD = d;
                }
            }

            if (bestGrad >= gradThresh && bestD > 0) {
                rightWidths.push_back(bestD);
            }
        }

        // Compute median per side, then global median
        auto getMedian = [](std::vector<int>& v) -> int {
            if (v.empty()) return -1;
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };

        int topMedian = getMedian(topWidths);
        int bottomMedian = getMedian(bottomWidths);
        int leftMedian = getMedian(leftWidths);
        int rightMedian = getMedian(rightWidths);

        // Collect all valid medians
        std::vector<int> sideMedians;
        if (topMedian > 0) sideMedians.push_back(topMedian);
        if (bottomMedian > 0) sideMedians.push_back(bottomMedian);
        if (leftMedian > 0) sideMedians.push_back(leftMedian);
        if (rightMedian > 0) sideMedians.push_back(rightMedian);

        if (!sideMedians.empty()) {
            measuredRailWidth = getMedian(sideMedians);
            // Clamp to reasonable bounds
            measuredRailWidth = std::max(5, std::min(measuredRailWidth, params.railBandWidth));
        }
    }

    // Step 2: Define table outer rectangle (felt bounds expanded by rail width)
    // Use symmetric edge-based clamping for consistency
    int left = std::max(0, feltBounds.x - measuredRailWidth);
    int right = std::min(rectifiedBgr.cols, feltBounds.x + feltBounds.width + measuredRailWidth);
    int top = std::max(0, feltBounds.y - measuredRailWidth);
    int bottom = std::min(rectifiedBgr.rows, feltBounds.y + feltBounds.height + measuredRailWidth);

    int outerX = left;
    int outerY = top;
    int outerW = right - left;
    int outerH = bottom - top;

    // Step 3: Create rail rectangles as bands between feltBounds and outerRect
    // This ensures rails are in table-local coordinates and include corners

    // Top rail: band from outer top to felt top, spanning outer width
    {
        int x = outerX;
        int y = outerY;
        int w = outerW;
        int h = feltBounds.y - outerY;

        if (h > 0 && w > 0) {
            cv::Rect topRailRect(x, y, w, h);

            result.railMasks[0] = cv::Mat::zeros(rectifiedBgr.size(), CV_8UC1);
            result.railMasks[0](topRailRect).setTo(255);
            result.numRailsDetected++;

            // Create outline as perfect rectangle
            result.railOutlines[0] = {
                topRailRect.tl(),
                cv::Point(topRailRect.br().x, topRailRect.tl().y),
                topRailRect.br(),
                cv::Point(topRailRect.tl().x, topRailRect.br().y)
            };
            result.railBoundingBoxes[0] = topRailRect;
        }
    }

    // Bottom rail: band from felt bottom to outer bottom, spanning outer width
    {
        int x = outerX;
        int y = feltBounds.y + feltBounds.height;
        int w = outerW;
        int h = (outerY + outerH) - y;

        if (h > 0 && w > 0 && y < rectifiedBgr.rows) {
            cv::Rect bottomRailRect(x, y, w, h);

            result.railMasks[2] = cv::Mat::zeros(rectifiedBgr.size(), CV_8UC1);
            result.railMasks[2](bottomRailRect).setTo(255);
            result.numRailsDetected++;

            // Create outline as perfect rectangle
            result.railOutlines[2] = {
                bottomRailRect.tl(),
                cv::Point(bottomRailRect.br().x, bottomRailRect.tl().y),
                bottomRailRect.br(),
                cv::Point(bottomRailRect.tl().x, bottomRailRect.br().y)
            };
            result.railBoundingBoxes[2] = bottomRailRect;
        }
    }

    // Left rail: band from outer left to felt left, spanning outer height
    {
        int x = outerX;
        int y = outerY;
        int w = feltBounds.x - outerX;
        int h = outerH;

        if (w > 0 && h > 0) {
            cv::Rect leftRailRect(x, y, w, h);

            result.railMasks[3] = cv::Mat::zeros(rectifiedBgr.size(), CV_8UC1);
            result.railMasks[3](leftRailRect).setTo(255);
            result.numRailsDetected++;

            // Create outline as perfect rectangle
            result.railOutlines[3] = {
                leftRailRect.tl(),
                cv::Point(leftRailRect.br().x, leftRailRect.tl().y),
                leftRailRect.br(),
                cv::Point(leftRailRect.tl().x, leftRailRect.br().y)
            };
            result.railBoundingBoxes[3] = leftRailRect;
        }
    }

    // Right rail: band from felt right to outer right, spanning outer height
    {
        int x = feltBounds.x + feltBounds.width;
        int y = outerY;
        int w = (outerX + outerW) - x;
        int h = outerH;

        if (w > 0 && h > 0 && x < rectifiedBgr.cols) {
            cv::Rect rightRailRect(x, y, w, h);

            result.railMasks[1] = cv::Mat::zeros(rectifiedBgr.size(), CV_8UC1);
            result.railMasks[1](rightRailRect).setTo(255);
            result.numRailsDetected++;

            // Create outline as perfect rectangle
            result.railOutlines[1] = {
                rightRailRect.tl(),
                cv::Point(rightRailRect.br().x, rightRailRect.tl().y),
                rightRailRect.br(),
                cv::Point(rightRailRect.tl().x, rightRailRect.br().y)
            };
            result.railBoundingBoxes[1] = rightRailRect;
        }
    }

    // Combine all rail masks
    for (int i = 0; i < 4; ++i) {
        if (!result.railMasks[i].empty()) {
            cv::bitwise_or(result.railMask, result.railMasks[i], result.railMask);
        }
    }

    // Success if at least one rail was detected
    result.ok = (result.numRailsDetected > 0);

    return result;
}

// Project rail detection results from rectified to perspective view
RailDetectionResult projectRailsToPerspective(
    const RailDetectionResult& rectifiedResult,
    const cv::Mat& Hinv,
    const cv::Size& perspectiveSize
) {
    RailDetectionResult perspectiveResult;
    perspectiveResult.ok = false;
    perspectiveResult.numRailsDetected = 0;
    
    // Validate inputs
    if (!rectifiedResult.ok || Hinv.empty() || Hinv.rows != 3 || Hinv.cols != 3) {
        return perspectiveResult;
    }
    
    // Project overall rail mask
    perspectiveResult.railMask = projectMaskToPerspective(rectifiedResult.railMask, Hinv, perspectiveSize);
    
    // Project individual rail masks
    perspectiveResult.railMasks.resize(4);
    perspectiveResult.railOutlines.resize(4);
    perspectiveResult.railBoundingBoxes.resize(4);
    
    for (int i = 0; i < 4; ++i) {
        if (!rectifiedResult.railMasks[i].empty()) {
            // Project mask
            perspectiveResult.railMasks[i] = projectMaskToPerspective(
                rectifiedResult.railMasks[i],
                Hinv,
                perspectiveSize
            );
            
            // Project outline
            if (!rectifiedResult.railOutlines[i].empty()) {
                perspectiveResult.railOutlines[i] = projectContourToPerspective(
                    rectifiedResult.railOutlines[i],
                    Hinv
                );
                
                // Compute bounding box in perspective space
                if (!perspectiveResult.railOutlines[i].empty()) {
                    perspectiveResult.railBoundingBoxes[i] = cv::boundingRect(perspectiveResult.railOutlines[i]);
                }
            }
            
            perspectiveResult.numRailsDetected++;
        }
    }
    
    perspectiveResult.ok = (perspectiveResult.numRailsDetected > 0);
    
    return perspectiveResult;
}

// Draw rail overlay on an image (in-place modification)
void drawRailOverlay(cv::Mat& img, const RailDetectionResult& result, const RailParams& params) {
    if (!result.ok || result.railMask.empty()) {
        return;
    }
    
    if (params.isFilled) {
        cv::Mat overlay = img.clone();
        // Create a 3-channel mask for blending
        std::vector<cv::Mat> maskChannels;
        cv::split(result.railMask, maskChannels);
        cv::Mat mask3ch;
        if (result.railMask.channels() == 1) {
            cv::merge(std::vector<cv::Mat>{maskChannels[0], maskChannels[0], maskChannels[0]}, mask3ch);
        } else {
            mask3ch = result.railMask.clone();
        }
        
        // Apply color overlay where mask is non-zero
        overlay.setTo(params.color, mask3ch);
        const double a = std::clamp(params.fillAlpha, 0, 255) / 255.0;
        cv::addWeighted(img, 1.0 - a, overlay, a, 0, img);
        
        // Draw outline for definition
        drawRailOutlines(img, result, params);
    } else {
        // Draw outline only
        drawRailOutlines(img, result, params);
    }
}

// Draw rail outlines on an image (in-place modification)
void drawRailOutlines(cv::Mat& img, const RailDetectionResult& result, const RailParams& params) {
    if (!result.ok) {
        return;
    }
    
    // Draw outlines for each detected rail
    for (int i = 0; i < 4; ++i) {
        if (!result.railOutlines[i].empty()) {
            std::vector<std::vector<cv::Point>> contours{result.railOutlines[i]};
            cv::drawContours(
                img,
                contours,
                -1,
                params.color,
                std::max(1, params.outlineThicknessPx)
            );
        }
    }
}

