#!/usr/bin/env python3
"""
Environment Verification Script
Verifies that all dependencies are properly installed and accessible.
"""

import sys
import os

def check_python():
    """Check Python version and environment"""
    print("🐍 Python Environment:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print(f"   Platform: {sys.platform}")
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    if conda_env:
        print(f"   Conda Environment: {conda_env}")
    else:
        print("   Conda Environment: Not detected")
    print()

def check_myo_import():
    """Check if myo module can be imported"""
    print("📦 Myo Module Check:")
    try:
        import myo
        print("   ✅ myo module imported successfully")
        
        # Try to get version info if available
        try:
            version = myo.__version__
            print(f"   Version: {version}")
        except:
            print("   Version: Unknown")
            
    except ImportError as e:
        print(f"   ❌ Failed to import myo: {e}")
        print("   💡 Try: conda activate myo_env")
        print("   💡 Or: pip install pylibmyo")
    except Exception as e:
        print(f"   ❌ Unexpected error importing myo: {e}")
    print()

def check_sdk_path():
    """Check if Myo SDK path exists"""
    print("🔧 SDK Path Check:")
    sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
    
    if os.path.exists(sdk_path):
        print(f"   ✅ SDK path exists: {sdk_path}")
        
        # Check for key files
        key_files = [
            "bin/myo64.dll",
            "include/myo/myo.hpp",
            "lib/myo64.lib"
        ]
        
        for file_path in key_files:
            full_path = os.path.join(sdk_path, file_path)
            if os.path.exists(full_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} (missing)")
    else:
        print(f"   ❌ SDK path not found: {sdk_path}")
    print()

def check_myo_connect():
    """Check if Myo Connect might be running"""
    print("🔗 Myo Connect Check:")
    print("   💡 Make sure Myo Connect is running")
    print("   💡 Make sure your Myo armband is paired")
    print("   💡 Check Windows Bluetooth settings")
    print()

def main():
    print("🔍 Myo Environment Verification")
    print("=" * 40)
    
    check_python()
    check_myo_import()
    check_sdk_path()
    check_myo_connect()
    
    print("✅ Environment check completed!")
    print("\n📋 Next Steps:")
    print("1. If all checks pass, run: python fixed_imu_test.py")
    print("2. If myo import fails, activate conda environment")
    print("3. If SDK path fails, check the path in the script")
    print("4. If Myo Connect issues, restart Myo Connect")

if __name__ == "__main__":
    main() 