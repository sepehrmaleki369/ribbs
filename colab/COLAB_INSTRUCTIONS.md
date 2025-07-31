# Google Colab Instructions

## Quick Setup

1. **Download the zip file**: `colab_project.zip` (created in the root directory)
2. **Upload to Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
3. **Create new notebook** or use the provided `vessel_distance_regression_colab.ipynb`

## Method 1: Using the Notebook

1. Upload `vessel_distance_regression_colab.ipynb` to Colab
2. Run each cell in order
3. Upload `colab_project.zip` when prompted

## Method 2: Manual Setup

1. Upload `colab_project.zip` to Colab
2. Extract the zip file
3. Install requirements: `!pip install -r requirements.txt`
4. Run training: `!python train_regression.py`
5. Run predictions: `!python show_best_regression_predictions.py`

## Files Included

- ✅ **Complete dataset** (drive/ folder)
- ✅ **Trained models** (checkpoints_regression/ folder)
- ✅ **All scripts** (train_regression.py, etc.)
- ✅ **Configuration files** (configs/ folder)
- ✅ **Core utilities** (core/ folder)
- ✅ **Model architectures** (models/ folder)

## Expected Results

- **Training time**: ~10-15 minutes on GPU
- **Validation MSE**: ~1000-1200
- **Validation MAE**: ~20-25
- **GPU acceleration**: 10-20x faster than CPU

## Tips

- Enable GPU in Colab: Runtime → Change runtime type → GPU
- The model will automatically use GPU if available
- All predictions will be saved in `predictions/` folder
- You can download results using Colab's file browser 