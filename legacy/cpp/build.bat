@echo off
REM Build script for Billiards Trainer
REM This script builds the project in Release configuration

echo Building Billiards Trainer...
echo.

REM Kill any running instances to prevent build lock
echo Checking for running instances...
taskkill /F /IM table_detector.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Closed running instance.
    timeout /t 1 /nobreak >nul
)

REM Create build directory if it doesn't exist
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

REM Navigate to build directory
cd build

REM Run CMake if needed (it will detect if it needs to reconfigure)
echo Running CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    cd ..
    exit /b 1
)

REM Build the project
echo.
echo Building project...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    cd ..
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable location: build\Release\table_detector.exe
cd ..

REM Launch the application
echo.
echo Launching application...
REM Copy runtime DLLs from vcpkg to the Release folder so the exe can find them
set VCPKG_BIN=C:\vcpkg\installed\x64-windows\bin
if exist "%VCPKG_BIN%" (
    echo Copying vcpkg runtime DLLs to build\Release...
    for %%f in (opencv_videoio4.dll opencv_imgcodecs4.dll opencv_imgproc4.dll opencv_core4.dll zlib1.dll jpeg62.dll libwebpdecoder.dll libwebp.dll libsharpyuv.dll libwebpdemux.dll libwebpmux.dll libpng16.dll tiff.dll liblzma.dll) do (
        if exist "%VCPKG_BIN%\%%f" (
            copy /Y "%VCPKG_BIN%\%%f" "build\Release\" >nul 2>&1 && echo Copied %%f || echo Failed to copy %%f
        ) else (
            echo MISSING IN VCPKG: %%f
        )
    )
) else (
    echo vcpkg bin folder not found at %VCPKG_BIN% - skipping DLL copy
)

REM Launch the application
echo.
echo Launching application...
start "" "build\Release\table_detector.exe"

