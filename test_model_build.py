import torch
import yaml
from pathlib import Path
import sys
import os

# --- Setup Project Root and YOLOv5 Path ---
# This script assumes it's in the project root (e.g., CNN_project_AI_course/)
# and the yolov5 directory is a subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parent
OUR_MODULE_DIR = PROJECT_ROOT / "Code_from_CHI_Xu"
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

if str(OUR_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(OUR_MODULE_DIR)) # 插入到最前面
    print(f"DEBUG: Prepended to sys.path for custom module files: {OUR_MODULE_DIR}")

# Add YOLOv5 root to sys.path to allow importing its modules
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))
    print(f"DEBUG: Added {YOLOV5_ROOT} to sys.path")

# Also add the directory containing 'Code_from_CHI_Xu' (which is PROJECT_ROOT)
# This is to help ensure 'from Code_from_CHI_Xu.our_modules import SEBlock'
# (if such an import is used inside yolov5/models/common.py) can be resolved
# when this script is run from the project root.
if str(PROJECT_ROOT) not in sys.path:
     sys.path.insert(0, str(PROJECT_ROOT))
     print(f"DEBUG: Added {PROJECT_ROOT} to sys.path for custom modules")


# Now try to import YOLOv5 modules.
# SEBlock should be made available to yolo.py's parse_model,
# typically by defining/importing it in yolov5/models/common.py
try:
    from models.yolo import Model  # YOLOv5's Model class
    from utils.general import check_yaml, LOGGER # YOLOv5's utility
    from utils.torch_utils import select_device
    print("SUCCESS: YOLOv5 and utility modules imported successfully.")
except ImportError as e:
    LOGGER.error(f"ERROR: Failed to import YOLOv5 modules. Ensure yolov5 directory is correct and in PYTHONPATH.")
    LOGGER.error(f"       This script expects to be in 'CNN_project_AI_course/' and 'yolov5' to be a subdirectory.")
    LOGGER.error(f"       Current working directory: {os.getcwd()}")
    LOGGER.error(f"       sys.path includes: {YOLOV5_ROOT} (should be listed)")
    LOGGER.error(f"ImportError: {e}")
    sys.exit(1)

def test_custom_model_creation(yaml_path_str: str, num_classes: int, channels: int = 3, device_str: str = 'cpu'):
    """
    Tests if a YOLOv5 model can be created from a given YAML configuration.
    """
    yaml_path = Path(yaml_path_str) # Convert string path to Path object
    LOGGER.info(f"\n--- Testing model creation with YAML: {yaml_path.name} ---")

    # Check if YAML file exists relative to PROJECT_ROOT if it's not an absolute path
    if not yaml_path.is_absolute() and not yaml_path.exists():
        yaml_path = PROJECT_ROOT / yaml_path_str # Try interpreting as relative to project root
        LOGGER.info(f"Interpreting YAML path relative to project root: {yaml_path}")


    if not yaml_path.exists():
        LOGGER.error(f"YAML configuration file not found at: {yaml_path_str} or {yaml_path}")
        LOGGER.error(f"Please ensure the path is correct. Current PWD: {os.getcwd()}")
        return None

    try:
        # 1. Check and load YAML file dictionary
        # check_yaml will prepend ROOT (yolov5 directory) if path is relative and not found
        # but since we are running from PROJECT_ROOT, we might need to be more explicit
        # Forcing check_yaml to look from PROJECT_ROOT perspective for our custom yaml
        cfg_yaml_path_checked = Path(check_yaml(str(yaml_path))) # Convert Path to str for check_yaml

        if not cfg_yaml_path_checked.exists(): # Double check after check_yaml
             LOGGER.error(f"check_yaml could not find the YAML file: {yaml_path}")
             return None

        LOGGER.info(f"Loading YAML from: {cfg_yaml_path_checked}")
        with open(cfg_yaml_path_checked, encoding="utf-8", errors="ignore") as f:
            cfg_dict = yaml.safe_load(f)  # model dict

        # 2. Override number of classes (nc) in the loaded dictionary
        # This ensures the test script uses the nc you specified,
        # and also helps verify if your YAML's nc is consistent.
        if 'nc' not in cfg_dict:
            LOGGER.warning(f"'nc' not found in {cfg_yaml_path_checked.name}. Setting nc={num_classes} from script.")
        elif cfg_dict['nc'] != num_classes:
            LOGGER.warning(
                f"Overriding {cfg_yaml_path_checked.name} nc={cfg_dict['nc']} with script's num_classes={num_classes}. "
                f"Ensure your YAML nc is correctly set to {num_classes} for actual training."
            )
        cfg_dict['nc'] = num_classes # This nc will be used by the Model constructor

        # 3. Select device
        device = select_device(device_str)
        LOGGER.info(f"Using device: {device}")

        # 4. Create model instance
        LOGGER.info("Attempting to create model instance...")
        # Model class from yolov5.models.yolo
        # It expects cfg (the dictionary loaded from yaml), ch (input channels), and nc (number of classes)
        model = Model(cfg=cfg_dict, ch=channels, nc=num_classes).to(device)
        model.eval()
        LOGGER.info(f"Model '{yaml_path.name}' created successfully!")

        # 5. (Optional but Recommended) Attempt a dummy forward pass
        LOGGER.info("Attempting a dummy forward pass...")
        img_size = 640 # Common YOLOv5 input size
        dummy_input = torch.randn(1, channels, img_size, img_size).to(device)
        with torch.no_grad():
            predictions = model(dummy_input) # For Model in eval mode, this is a list of Tensors (P3, P4, P5 outputs)
        LOGGER.info(f"Dummy forward pass successful.")

        LOGGER.info("Prediction output structure:")
        if isinstance(predictions, (list, tuple)):
            for i, p_tensor in enumerate(predictions):
                if isinstance(p_tensor, torch.Tensor):
                    LOGGER.info(f"  Output tensor {i} from detection head layer, shape: {p_tensor.shape}")
                else:
                    LOGGER.warning(f"  Output element {i} is not a Tensor, type: {type(p_tensor)}")
        elif isinstance(predictions, torch.Tensor):
            LOGGER.info(f"  Prediction output is a single tensor, shape: {predictions.shape}") # Less common for raw Model output
        else:
            LOGGER.warning(f"  Prediction output type not directly handled for shape logging: {type(predictions)}")

        return model

    except Exception as e:
        LOGGER.error(f"ERROR: Failed to create or test model from {yaml_path.name}.")
        LOGGER.error(f"Specific error: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # --- Configuration ---
    # ** IMPORTANT: Adjust these paths and parameters for your setup **

    # Path to your custom YOLOv5 model configuration YAML file
    # This path should be RELATIVE to this script (test_model_build.py)
    # OR an absolute path.
    # Given your input: "Code_from_CHI_Xu/yolov5s_with_se.yaml"
    # This implies the script is in CNN_project_AI_course/
    # and your yaml is in CNN_project_AI_course/Code_from_CHI_Xu/yolov5s_with_se.yaml
    CUSTOM_YAML_FILE_PATH = "Code_from_CHI_Xu/yolov5s_CBAM_only_neck.yaml"#每次测试改这个的路径，指向不同模型的配置文件

    NUMBER_OF_CLASSES_IN_YAML = 22  # ** This MUST match the nc value in your YAML for this test to be most meaningful **
                                  # Although the script will override cfg_dict['nc'], it's good for verification.

    INPUT_CHANNELS = 3     # Usually 3 for RGB images
    DEVICE_TO_USE = 'cpu'  # Use 'cpu' for build testing initially

    # --- Ensure SEBlock is importable by yolov5.models.common ---
    # This is a reminder. The actual import should happen in common.py as discussed.
    # If SEBlock is defined directly in common.py, no extra action here.
    # If SEBlock is in Code_from_CHI_Xu/our_modules.py, common.py needs to import it.

    LOGGER.info("--- Starting YOLOv5 Custom Model Build Test ---")
    LOGGER.info(f"Project Root (derived): {PROJECT_ROOT}")
    LOGGER.info(f"YOLOv5 Root (derived): {YOLOV5_ROOT}")
    LOGGER.info(f"Attempting to load custom YAML: {CUSTOM_YAML_FILE_PATH}")
    LOGGER.info(f"Expected number of classes (nc): {NUMBER_OF_CLASSES_IN_YAML}")


    custom_model_instance = test_custom_model_creation(
        yaml_path_str=CUSTOM_YAML_FILE_PATH,
        num_classes=NUMBER_OF_CLASSES_IN_YAML, # Pass the nc value
        channels=INPUT_CHANNELS,
        device_str=DEVICE_TO_USE
    )

    if custom_model_instance:
        LOGGER.info(f"\nSUCCESS: Custom YOLOv5 model defined in '{CUSTOM_YAML_FILE_PATH}' appears to be buildable and can perform a forward pass.")
        LOGGER.info("Next steps for Member A: ")
        LOGGER.info("  1. Carefully review your YAML, especially the 'from' indices in the 'head' section after inserting SEBlocks.")
        LOGGER.info("  2. Ensure SEBlock is correctly defined and imported in 'yolov5/models/common.py'.")
        LOGGER.info("  3. Commit these changes (common.py, your_custom_yaml.yaml, test_model_build.py) to your Git branch.")
        LOGGER.info("Next steps for Member C: ")
        LOGGER.info("  1. Pull these changes from Git.")
        LOGGER.info(f"  2. Use the custom YAML ('{CUSTOM_YAML_FILE_PATH}') to start training on the server/GPU with your custom dataset.")
    else:
        LOGGER.error(f"\nFAILURE: Custom YOLOv5 model build failed using '{CUSTOM_YAML_FILE_PATH}'.")
        LOGGER.error("Please check the error messages above. Common issues:")
        LOGGER.error("  - SEBlock class not found (check import in common.py or its definition).")
        LOGGER.error("  - Incorrect 'from' indices in the YAML 'head' section after modifying 'backbone'.")
        LOGGER.error("  - Channel mismatch between layers in YAML (ensure c1 of a module matches output of 'from' layer * width_multiple).")
        LOGGER.error("  - Typos in module names or parameters in YAML.")

    LOGGER.info("--- YOLOv5 Custom Model Build Test Finished ---")