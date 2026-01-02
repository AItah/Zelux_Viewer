# install:
- install:</br>
thorimagecam_v1.2.17_setup.exe
- thorlab package:</br>
python -m pip install "C:\...\thorlabs_tsi_camera_python_sdk_package.zip"
- read:</br>
Python_README.txt

- generate exe:</br>
pyinstaller --name ZeluxViewer --onefile --windowed --add-binary "dlls/64_lib/*.dll;dlls/64_lib" "C:\WC\SelfEmployee_wc\2025\STED with Nir\Code\Zelux_Viewer\src\main.py"

## Features
- Live view with start/stop controls, save/load image, and grayscale toggle.
- Image filtering (Low-pass / High-pass / Band-pass) applied to the live view, histogram, fits, and exported profiles.
- Exposure/gain control with dual exposure sliders (coarse and fine) plus set button and live value readout.
- Crosshair placement with live pixel/mm readout (mm relative to the cross as origin); clear cross button.
- Zoom/pan support (wheel zoom, middle-click fit-to-window, drag to pan when zoomed).
- Floating windows for histogram, fit controls, and main controls; window positions/sizes persist across runs.
- Fits and beam metrics:
- 2D Gaussian moment fit at the cross/center with ellipse overlay and waist ratio.
- SciPy Fit (Gaussian or Lorentzian chosen by AIC): auto horizontal/vertical profiles through the centroid, plots + metrics + overlay, and optional 2-point line fit (select two points then press SciPy Fit).
- Save packs results into a timestamped folder: displayed image (with overlays), fit plots (line/axis H/axis V), and cross-section CSV with raw profile data from the latest fits.
- Camera sources: native Thorlabs Zelux via SDK, Basler USB cameras via Pylon (`pypylon`), plus generic Windows/DirectShow cameras when OpenCV (`opencv-python`) is installed; a picker dialog appears when multiple cameras exist.

## Manual
- **Connect/Live**: Use Connect Camera to pick Zelux, Basler (Pylon), or any detected Windows camera (OpenCV required for generic). Then Start/Stop Live. Save/Load image available anytime.
- **View**: Scroll to zoom, drag to pan (when not fit-to-window), middle-click to fit. Fit button resets zoom/pan.
- **Crosshair**: Left-click to place anywhere (no snapping); Clear Cross removes. Pixel/mm readout shown in status bar; mm is relative to the current cross as the origin.
- **Exposure/Gain**: Adjust via sliders or spinbox; coarse and fine exposure sliders share the same value. Click Set to apply.
- **Windows**: Histogram/Fit toggles open floating windows; sizes/positions are remembered on restart.
- **Filters**: Choose None/Low-pass/High-pass/Band-pass and adjust low/high cutoffs (Nyquist fractions). Applied to image, histogram, and all fits/exports.
- **SciPy Fit**: Press SciPy Fit to auto-center, fit horizontal/vertical profiles (Gaussian vs. Lorentzian via AIC), and draw the waist ellipse. For a custom slice, click 2-Point Line Fit, pick two points (drag to refine), then press SciPy Fit again to fit that line and export the profile CSV + frame.**
- **Save**: Pick a filename; the app creates `<chosen_dir>/<timestamp>/` containing the displayed image with overlays, `line_plot.png`, `axis_h.png`, `axis_v.png` (when present), and `cross_sections.csv` with the raw profiles used in the latest fits.***
