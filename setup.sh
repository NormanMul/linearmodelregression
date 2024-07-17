#!/bin/bash
pip install --upgrade pip

 # Install libraries from requirements.txt, excluding NumPy
 pip install --no-deps -r requirements.txt

 # Install NumPy from the wheel file (Replace 'numpy‑1.24.3‑cp311‑cp311‑manylinux_2_17_x86_64.manylinux2014_x86_64.whl' with the actual filename)
 pip install numpy‑1.24.3‑cp311‑cp311‑manylinux_2_17_x86_64.manylinux2014_x86_64.whl 
