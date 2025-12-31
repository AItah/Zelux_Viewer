from pathlib import Path

p = Path(r"vendor\Scientific Camera Interfaces\SDK\Python Toolkit\thorlabs_tsi_camera_python_sdk_package.zip").resolve()
print(p.as_uri())
p = Path(r"C:\WC\SelfEmployee_wc\2025\STED with Nir\Code\BaslerTool\src\main.py").resolve()
print(p.as_uri())
