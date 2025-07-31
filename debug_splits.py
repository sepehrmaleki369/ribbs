import yaml
from core.general_dataset.splits import Split

def debug_splits():
    """Debug the splitter to see what modalities are found"""
    
    print("=== DEBUGGING SPLITS ===")
    
    # Load the regression dataset configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully")
    print(f"Base modalities: {config['base_modalities']}")
    print(f"Split config: {config['split_cfg']}")
    
    # Create splitter
    splitter = Split(config['split_cfg'], config['base_modalities'])
    
    # Try to get splits
    print("\nTrying to get splits...")
    try:
        train_split = splitter.get_split('train')
        print(f"Train split keys: {list(train_split.keys())}")
        for key, files in train_split.items():
            print(f"  {key}: {len(files)} files")
            if files:
                print(f"    First file: {files[0]}")
    except Exception as e:
        print(f"Error getting train split: {e}")
    
    # Try to build splits
    print("\nTrying to build splits...")
    try:
        splitter._build_splits()
        print("Splits built successfully")
    except Exception as e:
        print(f"Error building splits: {e}")

if __name__ == "__main__":
    debug_splits() 