#!/usr/bin/env python3
"""
Debug script to test GUI initialization step by step
"""

import sys
import os

print("🔍 Starting GUI debug...")

# Test 1: Basic imports
print("1. Testing basic imports...")
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, scrolledtext
    print("   ✅ tkinter imports OK")
except Exception as e:
    print(f"   ❌ tkinter import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    print("   ✅ matplotlib imports OK")
except Exception as e:
    print(f"   ❌ matplotlib import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✅ numpy import OK")
except Exception as e:
    print(f"   ❌ numpy import failed: {e}")
    sys.exit(1)

# Test 2: Myo import
print("2. Testing Myo import...")
try:
    import myo
    print("   ✅ myo import OK")
except Exception as e:
    print(f"   ❌ myo import failed: {e}")
    sys.exit(1)

# Test 3: Local module imports
print("3. Testing local module imports...")
try:
    from src.preprocessing import preprocess_emg, extract_features, extract_all_features
    print("   ✅ preprocessing import OK")
except Exception as e:
    print(f"   ❌ preprocessing import failed: {e}")
    sys.exit(1)

try:
    from src.model import train_model
    print("   ✅ model import OK")
except Exception as e:
    print(f"   ❌ model import failed: {e}")
    sys.exit(1)

try:
    from src.utils import get_sdk_path
    print("   ✅ utils import OK")
except Exception as e:
    print(f"   ❌ utils import failed: {e}")
    sys.exit(1)

# Test 4: Myo initialization
print("4. Testing Myo initialization...")
try:
    sdk_path = get_sdk_path()
    print(f"   SDK path: {sdk_path}")
    myo.init(sdk_path=sdk_path)
    print("   ✅ Myo initialization OK")
except Exception as e:
    print(f"   ❌ Myo initialization failed: {e}")
    sys.exit(1)

# Test 5: Basic GUI creation
print("5. Testing basic GUI creation...")
try:
    root = tk.Tk()
    root.withdraw()  # Hide the window
    print("   ✅ Basic Tkinter window creation OK")
    root.destroy()
except Exception as e:
    print(f"   ❌ Basic GUI creation failed: {e}")
    sys.exit(1)

print("🎉 All tests passed! The issue might be in the main GUI class.")
print("Try running the main GUI now...") 