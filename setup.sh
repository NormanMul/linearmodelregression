#!/bin/bash

pip install --upgrade pip

# Install NumPy and Pandas first, with specific versions
pip install numpy==1.24.3 pandas==1.5.3

# Install other libraries
pip install scikit-learn joblib
