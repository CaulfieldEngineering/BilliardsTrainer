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
start "" "build\Release\table_detector.exe"

