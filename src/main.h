#pragma once

// -------------------------------------------------------------------------------------------------
// main.h
//
// Purpose:
// - Hold the "shared" definitions that were previously at the top of `main.cpp`.
// - Keep `main.cpp` focused on implementation details.
//
// Notes:
// - This project is currently a single translation unit for the app, but having a header makes it
//   easier to split the app into multiple `.cpp` files later without copy/pasting core types.
// -------------------------------------------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <cmath> // std::lround

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX  // Prevent Windows.h from defining min/max macros
    #endif
    #include <windows.h>
    #include <commctrl.h>  // For trackbar controls
    #include <commdlg.h>   // For ChooseColor
    #include <uxtheme.h>   // For SetWindowTheme (modern-ish control styling)
    #include <algorithm>   // For std::max, std::min, std::sort

    // Link against common controls library (Win32 UI)
    #pragma comment(lib, "comctl32.lib")
    #pragma comment(lib, "comdlg32.lib")
    #pragma comment(lib, "uxtheme.lib")
#endif

#include "DiamondDetection/diamond_detection.h"
#include "FeltDetection/felt_detection.h"
#include "RailDetection/rail_detection.h"
#include "OrientationDetection/orientation_detector.h"

// Sidebar context types
enum class SidebarContext {
    None,
    Diamonds,
    Felt,
    Rail
};

// Which *top-level* sidebar page is currently shown.
// This is driven by the top menu (Debug / Display).
enum class SidebarPage {
    Debug,
    Display
};

// Selected input source sentinel values (match UIControls::selectedSource contract below)
inline constexpr int kSourceTestImage = -1;
inline constexpr int kSourceTestVideo = -2;

// Global variables for UI state
struct UIControls {
    bool showOverlay = true;  // Master overlay toggle (on by default)

    // Default all overlays ON so masks are visible immediately.
    bool showDiamonds = true;
    bool showFelt = true;
    bool showRail = true;

    // Orientation Mask: draws outer + inner quads bounding the rail mask, with labeled rail regions (L1, L2, S1, S2).
    bool showOrientation = false;

    bool showSidebar = true; // Debug sidebar toggle (on by default)
    bool sidebarCollapsed = false; // Sidebar collapsed state
    SidebarPage sidebarPage = SidebarPage::Debug; // Top-level sidebar page (default: Debug)
    SidebarContext sidebarContext = SidebarContext::Diamonds; // Current sidebar context (default to Diamonds)

    // Selected input source.
    //
    // Values:
    // - -1: Test Image (static `testImage.jpg`)
    // - -2: Test Video (looping `testVideo.mp4`)
    // - >= 0: camera index
    //
    // NOTE:
    // We intentionally use a single signed int here because it makes the Win32 combobox wiring
    // simple: we can store this value as CB_SETITEMDATA and treat all sources uniformly.
    // Default to Test Video so the app "does something" immediately without requiring a camera.
    int selectedSource = kSourceTestVideo;

    // Track which capture device/source is currently open in the legacy OpenCV-window path.
    // (Win32-hosted UI uses a background thread and does not rely on this.)
    int currentOpenedCamera = -1;

    // Global rendering/UX tuning.
    // 0 = no smoothing; 100 = very strong smoothing (nearly "sticky").
    // We apply this as smoothing of the *overlay delta* (not the base video)
    // so the raw video stays crisp while the overlay stabilizes.
    int smoothingPercent = 0;

    // Overlay parameters (defined in respective detection headers)
    DiamondDetectionParams diamondParams;
    FeltParams feltParams;
    RailParams railParams;
};

// Defined in `main.cpp`
extern UIControls uiControls;


