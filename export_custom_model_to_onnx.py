# 文件名: view_model_structure_to_onnx.py
# (将此脚本放在你的项目根目录 CNN_project_AI_course/ 下)

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys
import os

# --- Setup Project Root and YOLOv5 Path ---
# This script assumes it's in the project root (e.g., CNN_project_AI_course/)
# and the yolov5 directory is a subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

# Add YOLOv5 root to sys.path to allow importing its modules
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))
    print(f"DEBUG: Added {YOLOV5_ROOT} to sys.path")

# Also add the project root to sys.path to help find 'Code_from_CHI_Xu' if needed
# by imports within yolov5/models/common.py
if str(PROJECT_ROOT) not in sys.path:
     sys.path.insert(0, str(PROJECT_ROOT))
     print(f"DEBUG: Added {PROJECT_ROOT} to sys.path for custom modules (like Code_from_CHI_Xu)")

# Now try to import YOLOv5 modules.
# SEBlock (or your custom module) should be made available to yolo.py's parse_model,
# typically by defining/importing it in yolov5/models/common.py
try:
    from models.yolo import Model  # YOLOv5's Model class
    from utils.general import check_yaml, LOGGER # YOLOv5's utility
    from utils.torch_utils import select_device
    print("SUCCESS: YOLOv5 and utility modules imported successfully.")
except ImportError as e:
    print(f"ERROR: Failed to import YOLOv5 modules. Ensure yolov5 directory is correct and in PYTHONPATH.")
    print(f"       This script expects to be in 'CNN_project_AI_course/' and 'yolov5' to be a subdirectory.")
    print(f"       Current working directory: {os.getcwd()}")
    print(f"       sys.path includes: {YOLOV5_ROOT} (should be listed)")
    print(f"ImportError: {e}")
    sys.exit(1)

def export_structure_to_onnx(yaml_file_path_str: str,
                             onnx_output_path_str: str,
                             num_classes: int,
                             input_channels: int = 3,
                             img_size: int = 640,
                             opset: int = 12,
                             dynamic_axes: bool = True,
                             device_str: str = 'cpu'):
    """
    Builds a YOLOv5 model убийца_cfg (with initialized weights) and exports its structure to ONNX.
    """
    yaml_path = Path(yaml_file_path_str)
    onnx_output_path = Path(onnx_output_path_str)

    LOGGER.info(f"\n--- Exporting structure from YAML: {yaml_path.name} to ONNX: {onnx_output_path.name} ---")

    # Ensure YAML path is correctly interpreted (relative to project root if not absolute)
    if not yaml_path.is_absolute() and not yaml_path.exists():
        yaml_path = PROJECT_ROOT / yaml_path_str
        LOGGER.info(f"Interpreting YAML path relative to project root: {yaml_path}")

    if not yaml_path.exists():
        LOGGER.error(f"YAML configuration file not found at: {yaml_path_str} or {yaml_path}")
        return False

    try:
        # 1. Load YAML file dictionary
        cfg_yaml_path_checked = Path(check_yaml(str(yaml_path)))
        if not cfg_yaml_path_checked.exists():
             LOGGER.error(f"check_yaml could not find the YAML file: {yaml_path}")
             return False
        LOGGER.info(f"Loading YAML from: {cfg_yaml_path_checked}")
        with open(cfg_yaml_path_checked, encoding="utf-8", errors="ignore") as f:
            cfg_dict = yaml.safe_load(f)

        # 2. Set number of classes in the loaded dictionary
        cfg_dict['nc'] = num_classes
        LOGGER.info(f"Model will be built with nc={num_classes}")

        # 3. Select device
        device = select_device(device_str)
        LOGGER.info(f"Using device: {device} for model instantiation (weights will be random)")

        # 4. Create model instance (weights will be randomly initialized)
        LOGGER.info("Attempting to create model instance with initialized weights...")
        model = Model(cfg=cfg_dict, ch=input_channels, nc=num_classes).to(device)
        model.eval() # Set to evaluation mode
        LOGGER.info(f"Model '{yaml_path.name}' (structure only) created successfully!")

        # 5. Create a dummy input tensor
        dummy_input = torch.randn(1, input_channels, img_size, img_size).to(device)
        LOGGER.info(f"Created dummy input tensor of shape: {dummy_input.shape}")

        # 6. Set model attributes for ONNX export (YOLOv5 specific)
        if hasattr(model, 'model') and isinstance(model.model, nn.Sequential) and len(model.model) > 0:
            detect_layer = model.model[-1] # This is the Detect() layer instance
            if hasattr(detect_layer, 'export'):
                detect_layer.export = True # Important for ONNX export of Detect layer
                LOGGER.info("Set Detect().export = True for ONNX compatibility.")
            # If your Detect layer or model has a 'dynamic' attribute for grid generation:
            # if hasattr(detect_layer, 'dynamic'):
            #     detect_layer.dynamic = dynamic_axes

        # 7. Export to ONNX
        LOGGER.info(f"Exporting to ONNX: {onnx_output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_output_path), # torch.onnx.export expects a string path
            verbose=False, # Set to True for more ONNX export details
            opset_version=opset,
            input_names=['images'],
            output_names=['output'], # YOLOv5 model (raw, not AutoShape) typically has 1 output tensor from Detect layer in eval mode (if not training)
                                     # Or multiple if it's a list (P3, P4, P5 features before Detect processing)
                                     # For structure view, one output name is fine. Netron will show actual outputs.
            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},
                          'output': {0: 'batch', 1: 'anchors_x_predictions'}} if dynamic_axes else None
        )
        LOGGER.info(f"ONNX model structure saved to: {onnx_output_path}")
        return True

    except Exception as e:
        LOGGER.error(f"ERROR: Failed to export model structure from {yaml_path.name} to ONNX.")
        LOGGER.error(f"Specific error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # --- Configuration ---
    NUMBER_OF_CLASSES = 22  # ** YOUR DATASET'S CLASS COUNT **

    # --- 1. Export Baseline YOLOv5s Structure ---
    baseline_yaml_file = str(YOLOV5_ROOT / "models" / "yolov5s.yaml")
    baseline_onnx_output_file = str(PROJECT_ROOT / "yolov5s_baseline_structure.onnx")

    LOGGER.info(f"\n{'='*20} EXPORTING BASELINE MODEL STRUCTURE {'='*20}")
    export_structure_to_onnx(
        yaml_file_path_str=baseline_yaml_file,
        onnx_output_path_str=baseline_onnx_output_file,
        num_classes=NUMBER_OF_CLASSES, # Use your nc for apples-to-apples comparison of output layer
                                      # Or use 80 if you want to see the exact COCO-trained output layer structure
        img_size=640
    )

    # --- 2. Export Your Custom YOLOv5s with SEBlock Structure ---
    # Path to your custom YAML file
    custom_yaml_file = str(PROJECT_ROOT / "Code_from_CHI_Xu" / "yolov5s_with_se.yaml")
    custom_onnx_output_file = str(PROJECT_ROOT / "yolov5s_with_se_structure.onnx")

    LOGGER.info(f"\n{'='*20} EXPORTING CUSTOM SEBLOCK MODEL STRUCTURE {'='*20}")
    export_structure_to_onnx(
        yaml_file_path_str=custom_yaml_file,
        onnx_output_path_str=custom_onnx_output_file,
        num_classes=NUMBER_OF_CLASSES, # Your custom model is designed for nc=22
        img_size=640
    )

    LOGGER.info("\n--- ONNX Structure Export Script Finished ---")
    LOGGER.info("You can now open the generated .onnx files with Netron to view the model structures:")
    if Path(baseline_onnx_output_file).exists():
        LOGGER.info(f"  - Baseline: {baseline_onnx_output_file}")
    if Path(custom_onnx_output_file).exists():
        LOGGER.info(f"  - Custom (with SEBlock): {custom_onnx_output_file}")