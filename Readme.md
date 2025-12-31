# install:
- install 
thorimagecam_v1.2.17_setup.exe
- thorlab package:
python -m pip install "C:\...\thorlabs_tsi_camera_python_sdk_package.zip"
- read Python_README.txt

- generate exe
pyinstaller --onefile --windowed --add-binary "dlls/64_lib/*.dll;dlls/64_lib" "C:\WC\SelfEmployee_wc\2025\STED with Nir\Code\BaslerTool\src\main.py"

## Features
- Live view with start/stop controls, save/load image, and grayscale toggle.
- Exposure/gain control with dual exposure sliders (coarse and fine) plus set button and live value readout.
- Crosshair placement with live pixel/mm readout; clear cross button.
- Zoom/pan support (wheel zoom, middle-click fit-to-window, drag to pan when zoomed).
- Floating windows for histogram, fit controls, and main controls; window positions/sizes persist across runs.
- Gaussian fits:
  - 2D fit centered on cross or 360-fit center with ellipse overlay and waist ratio.
  - 2-point line fit with draggable line, profile plot, and Gaussian metrics (px/mm, FWHM, 1/e^2).
  - 360 fit: auto horizontal/vertical cross-sections through computed center, plots + metrics, overlay axes.

## Manual
- **Connect/Live**: Use Connect Camera, then Start/Stop Live. Save/Load image available anytime.
- **View**: Scroll to zoom, drag to pan (when not fit-to-window), middle-click to fit. Fit button resets zoom/pan.
- **Crosshair**: Left-click to place; Clear Cross removes. Pixel/mm readout shown in status bar.
- **Exposure/Gain**: Adjust via sliders or spinbox; coarse and fine exposure sliders share the same value. Click Set to apply.
- **Windows**: Histogram/Fit toggles open floating windows; sizes/positions are remembered on restart.
- **Line Fit**: Click 2-Point Line Fit, select two points, drag line/handles to refine, then Run Line Fit.
- **360 Fit**: After a line is selected, click Run 360 Fit to auto-fit horizontal/vertical axes and show ellipse/plots.
