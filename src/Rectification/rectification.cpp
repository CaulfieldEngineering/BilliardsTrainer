#include "rectification.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Helper function to compute intersection point of two lines
// Lines are represented as (x1, y1, x2, y2)
static cv::Point2f lineIntersection(const cv::Vec4i& line1, const cv::Vec4i& line2) {
    float x1 = static_cast<float>(line1[0]);
    float y1 = static_cast<float>(line1[1]);
    float x2 = static_cast<float>(line1[2]);
    float y2 = static_cast<float>(line1[3]);
    
    float x3 = static_cast<float>(line2[0]);
    float y3 = static_cast<float>(line2[1]);
    float x4 = static_cast<float>(line2[2]);
    float y4 = static_cast<float>(line2[3]);
    
    // Compute line equations: ax + by + c = 0
    float a1 = y2 - y1;
    float b1 = x1 - x2;
    float c1 = x2 * y1 - x1 * y2;
    
    float a2 = y4 - y3;
    float b2 = x3 - x4;
    float c2 = x4 * y3 - x3 * y4;
    
    // Solve for intersection
    float det = a1 * b2 - a2 * b1;
    if (std::abs(det) < 1e-6f) {
        // Lines are parallel, return invalid point
        return cv::Point2f(-1.0f, -1.0f);
    }
    
    float x = (b1 * c2 - b2 * c1) / det;
    float y = (a2 * c1 - a1 * c2) / det;
    
    return cv::Point2f(x, y);
}

// Helper function to compute angle of a line (in degrees)
static float lineAngle(const cv::Vec4i& line) {
    float dx = static_cast<float>(line[2] - line[0]);
    float dy = static_cast<float>(line[3] - line[1]);
    float angle = std::atan2(std::abs(dy), std::abs(dx)) * 180.0f / static_cast<float>(CV_PI);
    return angle;
}

RectificationResult rectifyTabletop(
    const cv::Mat& bgr,
    const cv::Mat& feltMask,
    const std::array<cv::Point2f,4>& cornersTLTRBRBL,
    float rectifyMarginScale,
    int rectifyPadPx
) {
    RectificationResult result;
    result.ok = false;
    result.srcQuad = cornersTLTRBRBL;

    // Validate inputs
    if (bgr.empty() || feltMask.empty()) {
        return result;
    }

    // Validate corners are finite
    bool allFinite = true;
    for (const auto& pt : cornersTLTRBRBL) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {
            allFinite = false;
            break;
        }
    }
    if (!allFinite) {
        return result;
    }

    // Validate parameters
    if (rectifyMarginScale < 1.0f || rectifyPadPx < 0) {
        return result;
    }

    // A) Temporarily disable centroid-based quad expansion - use detected quad directly
    // Store original quad as expanded (no expansion applied)
    result.expandedSrcQuad = cornersTLTRBRBL;

    // B) Compute destination width/height from the source quad
    // Corner order: TL, TR, BR, BL (indices 0, 1, 2, 3)
    const cv::Point2f& TL = cornersTLTRBRBL[0];
    const cv::Point2f& TR = cornersTLTRBRBL[1];
    const cv::Point2f& BR = cornersTLTRBRBL[2];
    const cv::Point2f& BL = cornersTLTRBRBL[3];
    
    // Compute edge lengths using Euclidean distance
    const float wTop = cv::norm(TR - TL);
    const float wBot = cv::norm(BR - BL);
    const float hLeft = cv::norm(BL - TL);
    const float hRight = cv::norm(BR - TR);

    // Determine natural dimensions (max of corresponding edges)
    float naturalW = std::max(wTop, wBot);
    float naturalH = std::max(hLeft, hRight);

    // Analyze pool table orientation and force 2:1 portrait aspect ratio
    // Pool tables are 2:1 ratio (length:width). We want to display in portrait mode.
    // Portrait means height > width, so we want height = 2 * width

    // Determine which dimension represents the "long" side of the table
    // If naturalW > naturalH, the table is oriented horizontally in the source
    // If naturalH > naturalW, the table is oriented vertically in the source

    float dstW, dstH;
    if (naturalW > naturalH) {
        // Table is horizontal in source -> we need to rotate to portrait
        // The long dimension (naturalW) becomes our height in portrait mode
        dstH = naturalW;  // Long dimension becomes height
        dstW = dstH / 2.0f;  // Width is half of height for 2:1 portrait
    } else {
        // Table is vertical in source -> already portrait-ish
        // The long dimension (naturalH) is our height
        dstH = naturalH;  // Long dimension is already height
        dstW = dstH / 2.0f;  // Width is half of height for 2:1 portrait
    }
    
    // Calculate destination size with padding
    const int baseDstW = static_cast<int>(std::round(dstW)) + 2 * rectifyPadPx;
    const int baseDstH = static_cast<int>(std::round(dstH)) + 2 * rectifyPadPx;
    result.dstSize = cv::Size(baseDstW, baseDstH);

    // Define destination corners inset by padding
    const float w = static_cast<float>(result.dstSize.width);
    const float h = static_cast<float>(result.dstSize.height);
    const float padF = static_cast<float>(rectifyPadPx);
    result.dstQuad = {
        cv::Point2f(padF, padF),                           // TL
        cv::Point2f(w - 1.0f - padF, padF),                // TR
        cv::Point2f(w - 1.0f - padF, h - 1.0f - padF),    // BR
        cv::Point2f(padF, h - 1.0f - padF)                 // BL
    };

    // Compute homography matrix using source quad (no expansion)
    result.H = cv::getPerspectiveTransform(result.expandedSrcQuad, result.dstQuad);

    // Check if H is valid (not degenerate)
    if (result.H.empty() || result.H.rows != 3 || result.H.cols != 3) {
        return result;
    }

    // Check for degenerate transformation (determinant too small)
    // For a 3x3 homography matrix, check the determinant of the 2x2 upper-left submatrix
    // which represents the affine part of the transformation
    double det = result.H.at<double>(0, 0) * result.H.at<double>(1, 1) - result.H.at<double>(0, 1) * result.H.at<double>(1, 0);
    if (std::abs(det) < 1e-6) {
        return result;
    }

    // Compute inverse homography
    result.Hinv = result.H.inv();
    if (result.Hinv.empty() || result.Hinv.rows != 3 || result.Hinv.cols != 3) {
        return result;
    }

    // Warp the BGR image using linear interpolation
    cv::Mat initialRectifiedBgr;
    cv::warpPerspective(
        bgr,
        initialRectifiedBgr,
        result.H,
        result.dstSize,
        cv::INTER_LINEAR
    );

    // Warp the felt mask first (needed for refinement)
    cv::Mat initialRectifiedFeltMask;
    cv::warpPerspective(
        feltMask,
        initialRectifiedFeltMask,
        result.H,
        result.dstSize,
        cv::INTER_NEAREST
    );

    // Refinement step: detect rail edges in the roughly-rectified image
    // and compute a correction homography to achieve perfect rectangular shape
    cv::Mat H_correction = cv::Mat::eye(3, 3, CV_64F);
    bool refinementSuccess = false;
    
    if (!initialRectifiedBgr.empty() && !initialRectifiedFeltMask.empty()) {
        // Create a boundary band mask: narrow band (30-50 pixels) around the felt mask boundary
        // This excludes interior objects like cue sticks and balls
        const int bandWidth = 40;  // pixels
        cv::Mat boundaryBand = cv::Mat::zeros(initialRectifiedFeltMask.size(), CV_8UC1);
        
        // Create boundary band by dilating and eroding the felt mask
        // The band is the region between the outer and inner boundaries
        cv::Mat dilated, eroded;
        cv::dilate(initialRectifiedFeltMask, dilated, cv::Mat(), cv::Point(-1, -1), bandWidth);
        cv::erode(initialRectifiedFeltMask, eroded, cv::Mat(), cv::Point(-1, -1), bandWidth);
        
        // Band is pixels in dilated but not in eroded (difference operation)
        // This creates a band around the boundary of the felt mask
        cv::subtract(dilated, eroded, boundaryBand);
        
        // Convert to grayscale
        cv::Mat gray;
        if (initialRectifiedBgr.channels() == 3) {
            cv::cvtColor(initialRectifiedBgr, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = initialRectifiedBgr.clone();
        }
        
        // Apply Canny edge detection only within the boundary band
        cv::Mat edges = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat grayMasked;
        gray.copyTo(grayMasked, boundaryBand);
        cv::Canny(grayMasked, edges, 50, 150, 3);
        // Mask the edges to only include the boundary band
        cv::bitwise_and(edges, boundaryBand, edges);
        
        // Use Hough line detection with stricter parameters
        // Increased minimum line length (100+ pixels) and higher vote threshold
        std::vector<cv::Vec4i> lines;
        int minLineLength = 100;
        int maxLineGap = 20;
        int voteThreshold = 80;  // Increased from default 50
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, voteThreshold, minLineLength, maxLineGap);
        
        if (lines.size() >= 4) {
            // Filter lines into horizontal (top/bottom rails) and vertical (left/right rails)
            // Since image is roughly rectified, horizontal lines should have small angle (< 15 degrees)
            // and vertical lines should have large angle (> 75 degrees)
            std::vector<cv::Vec4i> horizontalLines;
            std::vector<cv::Vec4i> verticalLines;
            
            for (const auto& line : lines) {
                float angle = lineAngle(line);
                if (angle < 15.0f) {
                    horizontalLines.push_back(line);
                } else if (angle > 75.0f) {
                    verticalLines.push_back(line);
                }
            }
            
            // Need at least 2 horizontal and 2 vertical lines
            if (horizontalLines.size() >= 2 && verticalLines.size() >= 2) {
                // Get approximate felt bounds to define expected band regions
                cv::Rect feltBounds = cv::boundingRect(initialRectifiedFeltMask);
                float feltTop = static_cast<float>(feltBounds.y);
                float feltBottom = static_cast<float>(feltBounds.y + feltBounds.height);
                float feltLeft = static_cast<float>(feltBounds.x);
                float feltRight = static_cast<float>(feltBounds.x + feltBounds.width);
                
                // Cluster horizontal lines by position and select median for top/bottom
                // Top rail: lines near the top of the felt (within bandWidth of top)
                // Bottom rail: lines near the bottom of the felt (within bandWidth of bottom)
                std::vector<float> topLineYs, bottomLineYs;
                for (const auto& line : horizontalLines) {
                    float y = (static_cast<float>(line[1]) + static_cast<float>(line[3])) / 2.0f;
                    if (std::abs(y - feltTop) < static_cast<float>(bandWidth * 2)) {
                        topLineYs.push_back(y);
                    } else if (std::abs(y - feltBottom) < static_cast<float>(bandWidth * 2)) {
                        bottomLineYs.push_back(y);
                    }
                }
                
                // Cluster vertical lines by position and select median for left/right
                // Left rail: lines near the left of the felt
                // Right rail: lines near the right of the felt
                std::vector<float> leftLineXs, rightLineXs;
                for (const auto& line : verticalLines) {
                    float x = (static_cast<float>(line[0]) + static_cast<float>(line[2])) / 2.0f;
                    if (std::abs(x - feltLeft) < static_cast<float>(bandWidth * 2)) {
                        leftLineXs.push_back(x);
                    } else if (std::abs(x - feltRight) < static_cast<float>(bandWidth * 2)) {
                        rightLineXs.push_back(x);
                    }
                }
                
                // Need at least one line in each cluster
                if (topLineYs.size() >= 1 && bottomLineYs.size() >= 1 && 
                    leftLineXs.size() >= 1 && rightLineXs.size() >= 1) {
                    
                    // Get median positions for each side
                    std::sort(topLineYs.begin(), topLineYs.end());
                    std::sort(bottomLineYs.begin(), bottomLineYs.end());
                    std::sort(leftLineXs.begin(), leftLineXs.end());
                    std::sort(rightLineXs.begin(), rightLineXs.end());
                    
                    float topY = topLineYs[topLineYs.size() / 2];
                    float bottomY = bottomLineYs[bottomLineYs.size() / 2];
                    float leftX = leftLineXs[leftLineXs.size() / 2];
                    float rightX = rightLineXs[rightLineXs.size() / 2];
                    
                    // Find lines closest to these median positions
                    cv::Vec4i topLine, bottomLine, leftLine, rightLine;
                    float minTopDist = std::numeric_limits<float>::max();
                    float minBottomDist = std::numeric_limits<float>::max();
                    float minLeftDist = std::numeric_limits<float>::max();
                    float minRightDist = std::numeric_limits<float>::max();
                    
                    for (const auto& line : horizontalLines) {
                        float y = (static_cast<float>(line[1]) + static_cast<float>(line[3])) / 2.0f;
                        float distTop = std::abs(y - topY);
                        float distBottom = std::abs(y - bottomY);
                        if (distTop < minTopDist) {
                            minTopDist = distTop;
                            topLine = line;
                        }
                        if (distBottom < minBottomDist) {
                            minBottomDist = distBottom;
                            bottomLine = line;
                        }
                    }
                    
                    for (const auto& line : verticalLines) {
                        float x = (static_cast<float>(line[0]) + static_cast<float>(line[2])) / 2.0f;
                        float distLeft = std::abs(x - leftX);
                        float distRight = std::abs(x - rightX);
                        if (distLeft < minLeftDist) {
                            minLeftDist = distLeft;
                            leftLine = line;
                        }
                        if (distRight < minRightDist) {
                            minRightDist = distRight;
                            rightLine = line;
                        }
                    }
                    
                    // Compute four intersection points
                    cv::Point2f topLeft = lineIntersection(topLine, leftLine);
                    cv::Point2f topRight = lineIntersection(topLine, rightLine);
                    cv::Point2f bottomRight = lineIntersection(bottomLine, rightLine);
                    cv::Point2f bottomLeft = lineIntersection(bottomLine, leftLine);
                    
                    // Validate all intersections are valid (within image bounds)
                    bool allValid = true;
                    std::array<cv::Point2f, 4> detectedQuad = {topLeft, topRight, bottomRight, bottomLeft};
                    
                    for (const auto& pt : detectedQuad) {
                        if (pt.x < 0 || pt.y < 0 || 
                            pt.x >= result.dstSize.width || pt.y >= result.dstSize.height ||
                            !std::isfinite(pt.x) || !std::isfinite(pt.y)) {
                            allValid = false;
                            break;
                        }
                    }
                    
                    if (allValid) {
                        // Compute bounding box of detected quad to determine size
                        float minX = std::min({topLeft.x, topRight.x, bottomRight.x, bottomLeft.x});
                        float maxX = std::max({topLeft.x, topRight.x, bottomRight.x, bottomLeft.x});
                        float minY = std::min({topLeft.y, topRight.y, bottomRight.y, bottomLeft.y});
                        float maxY = std::max({topLeft.y, topRight.y, bottomRight.y, bottomLeft.y});
                        
                        float detectedWidth = maxX - minX;
                        float detectedHeight = maxY - minY;
                        
                        // Validation: check aspect ratio is within 20% of 2:1
                        float detectedAspectRatio = detectedWidth / detectedHeight;
                        float targetAspectRatio = 2.0f;
                        float aspectRatioError = std::abs(detectedAspectRatio - targetAspectRatio) / targetAspectRatio;
                        
                        if (aspectRatioError <= 0.20f && detectedWidth > 0 && detectedHeight > 0) {
                            // Define target rectangle with 2:1 aspect ratio (standard pool table)
                            // Use the average of detected width and height to maintain scale
                            float avgSize = (detectedWidth + detectedHeight) / 2.0f;
                            
                            // For 2:1 aspect ratio (width:height = 2:1), width = 2 * height
                            float targetHeight = avgSize;
                            float targetWidth = targetHeight * 2.0f;  // width = 2 * height for 2:1 ratio
                            
                            // Ensure target fits within reasonable bounds of detected quad
                            if (targetWidth > detectedWidth * 1.5f) {
                                targetWidth = detectedWidth * 1.5f;
                                targetHeight = targetWidth / 2.0f;
                            }
                            if (targetHeight > detectedHeight * 1.5f) {
                                targetHeight = detectedHeight * 1.5f;
                                targetWidth = targetHeight * 2.0f;
                            }
                            
                            // Center the target rectangle in the destination space
                            float centerX = (minX + maxX) / 2.0f;
                            float centerY = (minY + maxY) / 2.0f;
                            
                            // Ensure target rectangle fits within image bounds
                            float halfWidth = targetWidth / 2.0f;
                            float halfHeight = targetHeight / 2.0f;
                            
                            // Clamp to image bounds with some margin
                            float margin = 10.0f;
                            centerX = std::max(halfWidth + margin, std::min(static_cast<float>(result.dstSize.width) - halfWidth - margin, centerX));
                            centerY = std::max(halfHeight + margin, std::min(static_cast<float>(result.dstSize.height) - halfHeight - margin, centerY));
                            
                            std::array<cv::Point2f, 4> targetQuad = {
                                cv::Point2f(centerX - halfWidth, centerY - halfHeight),  // TL
                                cv::Point2f(centerX + halfWidth, centerY - halfHeight),  // TR
                                cv::Point2f(centerX + halfWidth, centerY + halfHeight),  // BR
                                cv::Point2f(centerX - halfWidth, centerY + halfHeight)   // BL
                            };
                            
                            // Compute correction homography from detected quad to target quad
                            H_correction = cv::getPerspectiveTransform(detectedQuad, targetQuad);
                            
                            // Validate correction homography
                            if (!H_correction.empty() && H_correction.rows == 3 && H_correction.cols == 3) {
                                double det_corr = H_correction.at<double>(0, 0) * H_correction.at<double>(1, 1) - 
                                                 H_correction.at<double>(0, 1) * H_correction.at<double>(1, 0);
                                if (std::abs(det_corr) > 1e-6) {
                                    refinementSuccess = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Compose homographies: H_refined = H_correction * H_original
    // If refinement failed, H_correction is identity, so H_refined = H_original
    cv::Mat H_refined = H_correction * result.H;
    
    // Re-warp from source image using composed homography
    cv::warpPerspective(
        bgr,
        result.rectifiedBgr,
        H_refined,
        result.dstSize,
        cv::INTER_LINEAR
    );
    
    // Update result with refined homography
    result.H = H_refined;
    result.Hinv = result.H.inv();
    if (result.Hinv.empty() || result.Hinv.rows != 3 || result.Hinv.cols != 3) {
        return result;
    }

    // Warp the felt mask using nearest neighbor interpolation (preserve binary mask)
    // Use the refined homography
    cv::warpPerspective(
        feltMask,
        result.rectifiedFeltMask,
        result.H,
        result.dstSize,
        cv::INTER_NEAREST
    );

    // Validate outputs
    if (result.rectifiedBgr.empty() || result.rectifiedFeltMask.empty()) {
        return result;
    }

    // Success
    result.ok = true;
    return result;
}

