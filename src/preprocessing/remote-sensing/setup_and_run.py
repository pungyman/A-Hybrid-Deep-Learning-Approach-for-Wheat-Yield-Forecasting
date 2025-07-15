#!/usr/bin/env python3
"""
Setup and run script for NDVI/EVI extraction from Google Earth Engine
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def authenticate_ee():
    """Authenticate with Google Earth Engine"""
    print("\nAuthenticating with Google Earth Engine...")
    print("This will open a browser window for authentication.")
    
    try:
        subprocess.check_call(["earthengine", "authenticate"])
        print("✓ Earth Engine authentication successful!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error authenticating with Earth Engine: {e}")
        print("Please make sure you have a Google Earth Engine account.")
        return False
    except FileNotFoundError:
        print("✗ Earth Engine CLI not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
            subprocess.check_call(["earthengine", "authenticate"])
            print("✓ Earth Engine authentication successful!")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    return True

def check_shapefile():
    """Check if shapefile exists"""
    shapefile_path = "shapefiles/DISTRICT_BOUNDARY.shp"
    if os.path.exists(shapefile_path):
        print(f"✓ Shapefile found: {shapefile_path}")
        return True
    else:
        print(f"✗ Shapefile not found: {shapefile_path}")
        print("Please make sure the district boundary shapefile is in the current directory.")
        return False

def run_extraction():
    """Run the NDVI/EVI extraction"""
    print("\nRunning NDVI/EVI extraction...")
    try:
        subprocess.check_call([sys.executable, "extract_ndvi_evi.py"])
        print("✓ Extraction completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running extraction: {e}")
        return False
    return True

def main():
    """Main setup and run function"""
    print("NDVI/EVI Extraction Setup")
    print("=" * 30)
    
    # Step 1: Install requirements
    if not install_requirements():
        return
    
    # Step 2: Check shapefile
    if not check_shapefile():
        return
    
    # Step 3: Authenticate with Earth Engine
    if not authenticate_ee():
        return
    
    # Step 4: Run extraction
    print("\nReady to run extraction!")
    user_input = input("Do you want to run the extraction now? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        run_extraction()
    else:
        print("Setup complete! You can run the extraction later with:")
        print("python extract_ndvi_evi.py")

if __name__ == "__main__":
    main() 