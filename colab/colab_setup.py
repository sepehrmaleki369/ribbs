# Google Colab Setup Script
# Run this cell to set up the project

import os
import zipfile
from google.colab import files

# Upload the project zip file
print("Please upload the colab_project.zip file...")
uploaded = files.upload()

# Extract the project
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"✅ Extracted {filename}")

# Install requirements
print("Installing requirements...")
!pip install -r requirements.txt

# Verify setup
print("\n=== PROJECT SETUP COMPLETE ===")
print("Available files:")
!ls -la

print("\nAvailable directories:")
!ls -la | grep "^d"

print("\n✅ Project ready for training!") 