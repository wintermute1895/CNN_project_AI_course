# obstacle_detector.py

import torch
import numpy as np
import cv2  # OpenCV 用于图像处理
from pathlib import Path  # 用于路径操作

# --- 动态调整 sys.path 以确保能找到 yolov5 模块 ---
# 这个脚本可能被其他脚本从不同位置导入，所以需要确保yolov5路径的稳定性
try:
    FILE_SCRIPT = Path(__file__).resolve()  # 当前脚本的绝对路径
    # 假设 obstacle_detector.py 与 yolov5 文件夹在同一个父目录下 (例如项目根目录)
    PROJECT_ROOT_FROM_SCRIPT = FILE_SCRIPT.parents[0]  # 获取当前脚本所在目录
    YOLOV5_ROOT_FROM_SCRIPT = PROJECT_ROOT_FROM_SCRIPT / "yolov5"

    if str(YOLOV5_ROOT_FROM_SCRIPT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT_FROM_SCRIPT))  # 优先加载这个路径下的yolov5
        # print(f"DEBUG (obstacle_detector): Added {YOLOV5_ROOT_FROM_SCRIPT} to sys.path")
except NameError:  # 如果直接运行此脚本 (不是作为模块导入)，__file__ 可能未定义
    # 这种情况下，我们假设 yolov5 目录与脚本在同一级，或者已经通过其他方式在sys.path中
    # 或者，如果这是一个会被其他脚本导入的模块，那么导入它的脚本应该负责设置好sys.path
    # print("DEBUG (obstacle_detector): __file__ not defined, assuming yolov5 is in sys.path or relative.")
    pass  # 继续尝试导入，如果失败，后续会报错

try:
    from models.experimental import attempt_load  # YOLOv5 模型加载函数
    from utils.general import non_max_suppression, LOGGER  # NMS 和日志
    from utils.torch_utils import select_device
    from utils.dataloaders import letterbox  # YOLOv5 v7.0 推荐的 letterbox 位置
except ImportError as e:
    print(f"CRITICAL ERROR in obstacle_detector.py: Failed to import YOLOv5 specific modules.")
    print(f"  Please ensure the 'yolov5' directory is correctly placed relative to this script or in PYTHONPATH.")
    print(f"  Current sys.path relevant entries:")
    # for p_entry in sys.path:
    #     if "yolov5" in str(p_entry).lower() or "cnn_project" in str(p_entry).lower():
    #         print(f"    {p_entry}")
    print(f"  ImportError: {e}")
    raise  # 重新抛出异常，因为没有这些模块无法工作
# ----------------------------------------------------

# --- 全局配置和模型加载 ---
# TODO: 【你需要根据你的实际情况修改以下路径和参数】
MODEL_WEIGHTS_PATH = 'yolov5s.pt'  # 默认使用官方yolov5s.pt，它会被自动下载（如果网络通畅）
# 或者替换为你的自定义模型权重路径, 例如: 'Code_from_CHI_Xu/weights/best.pt'
MODEL_CFG_PATH = "yolov5/models/yolov5s.yaml" # 如果你的模型是自定义结构，可能需要提供对应的.yaml配置文件路径
# 例如: 'Code_from_CHI_Xu/yolov5s_v7_cbam_only_neck.yaml'
# 对于官方yolov5s.pt，通常不需要cfg，attempt_load会处理
DEVICE_STR = 'cpu'  # 推理设备: 'cpu', '0', '1', '0,1' (for cuda device IDs)
CONF_THRESHOLD = 0.25  # NMS 置信度阈值
IOU_THRESHOLD = 0.45  # NMS IoU 阈值
DEFAULT_IMG_SIZE = 640  # 模型推理时期望的图像尺寸 (通常是640 for YOLOv5s)

# --- 全局模型变量 (延迟加载) ---
MODEL = None
DEVICE = None
CLASS_NAMES = None
MODEL_STRIDE = 32  # 默认 stride


def _load_model():
    """
    内部函数，用于加载YOLOv5模型。只在第一次调用检测函数时执行。
    """
    global MODEL, DEVICE, CLASS_NAMES, MODEL_STRIDE  # 声明我们要修改全局变量

    if MODEL is not None:  # 如果模型已加载，则直接返回
        return

    try:
        LOGGER.info(f"Loading YOLOv5 model from: {MODEL_WEIGHTS_PATH} (cfg: {MODEL_CFG_PATH}) on device: {DEVICE_STR}")
        DEVICE = select_device(DEVICE_STR)

        # attempt_load 会尝试从 .pt 文件中加载模型和结构
        # fuse=True 可以融合 Conv+BN 加速推理，但有时可能导致数值问题，fuse=False 更安全
        model_loaded = attempt_load(MODEL_WEIGHTS_PATH, device=DEVICE, fuse=True)  # inplace=True,

        # 如果是 DataParallel 包装的模型，获取其 .module
        MODEL = model_loaded.module if hasattr(model_loaded, 'module') else model_loaded
        MODEL.eval()  # 设置为评估模式

        MODEL_STRIDE = int(MODEL.stride.max()) if hasattr(MODEL, 'stride') and MODEL.stride is not None else 32

        # 获取类别名称
        if hasattr(MODEL, 'names') and MODEL.names:
            CLASS_NAMES = MODEL.names
            if isinstance(CLASS_NAMES, list) and all(isinstance(name, str) for name in CLASS_NAMES):
                # 如果是列表，转换为YOLOv5常用的字典格式 {id: name}
                CLASS_NAMES = {i: name for i, name in enumerate(CLASS_NAMES)}
            elif not isinstance(CLASS_NAMES, dict):
                LOGGER.warning(f"Model 'names' attribute is not a dict or list of strings. Type: {type(CLASS_NAMES)}")
                CLASS_NAMES = None  # 重置为None，以便使用备用方案

        if CLASS_NAMES is None:
            LOGGER.warning(
                f"Model at '{MODEL_WEIGHTS_PATH}' does not contain 'names'. Attempting to load from CFG or using default.")
            if MODEL_CFG_PATH and Path(MODEL_CFG_PATH).exists():
                import yaml
                try:
                    with open(MODEL_CFG_PATH, 'r', encoding='utf-8') as f:
                        cfg_data = yaml.safe_load(f)
                    CLASS_NAMES = cfg_data.get('names')
                    if isinstance(CLASS_NAMES, list):  # 确保是字典
                        CLASS_NAMES = {i: name for i, name in enumerate(CLASS_NAMES)}
                    if not CLASS_NAMES:  # 如果yaml里也没有或为空
                        nc_from_cfg = cfg_data.get('nc', 80)
                        CLASS_NAMES = {i: f'class_{i}' for i in range(nc_from_cfg)}
                        LOGGER.info(f"Using default class names for nc={nc_from_cfg} from CFG.")
                except Exception as e_yaml:
                    LOGGER.error(f"Error loading names from CFG '{MODEL_CFG_PATH}': {e_yaml}")
                    CLASS_NAMES = {i: f'class_{i}' for i in range(80)}  # 最终备用
                    LOGGER.info("Using generic default class names (80 classes).")
            else:  # 如果CFG路径也没有，用通用备用
                CLASS_NAMES = {i: f'class_{i}' for i in range(80)}  # 通用备用
                LOGGER.info("Using generic default class names (80 classes). Please ensure this matches your model.")

        LOGGER.info(
            f"Model loaded successfully. Stride: {MODEL_STRIDE}, Class Names: {CLASS_NAMES if CLASS_NAMES else 'Not properly loaded'}")

    except Exception as e:
        LOGGER.error(f"CRITICAL ERROR: Failed to load YOLOv5 model in _load_model().")
        LOGGER.error(f"  WEIGHTS_PATH: {MODEL_WEIGHTS_PATH}")
        LOGGER.error(f"  CFG_PATH (if used by attempt_load internally or for names): {MODEL_CFG_PATH}")
        LOGGER.error(f"  Error: {e}", exc_info=True)  # exc_info=True 会打印堆栈跟踪
        MODEL = None  # 确保模型状态是None，以便后续调用知道加载失败
        raise  # 重新抛出异常，让调用者知道失败了


def detect_obstacles_in_image(image_bgr_numpy,
                              img_size=DEFAULT_IMG_SIZE,
                              conf_thres=CONF_THRESHOLD,
                              iou_thres=IOU_THRESHOLD):
    """
    对输入的单张图像进行障碍物检测。

    Args:
        image_bgr_numpy (numpy.ndarray): 输入的图像，格式为 BGR，形状为 (H, W, C)。
        img_size (int): 模型推理时期望的图像尺寸 (正方形的一边长度)。
        conf_thres (float): NMS 的置信度阈值。
        iou_thres (float): NMS 的 IoU 阈值。

    Returns:
        tuple: (detections_list, image_height, image_width)
            detections_list (list): 检测到的障碍物列表。如果无检测，则为空列表[]。
            image_height (int): 原始输入图像的高度。
            image_width (int): 原始输入图像的宽度。
    """
    if MODEL is None:  # 检查模型是否已加载，如果未加载则尝试加载
        try:
            _load_model()
            if MODEL is None:  # 如果 _load_model 内部加载失败，MODEL 仍然是 None
                LOGGER.error("Model could not be loaded. Cannot perform detection.")
                return [], image_bgr_numpy.shape[0], image_bgr_numpy.shape[1]
        except Exception as e_load:  # 捕获 _load_model 可能抛出的异常
            LOGGER.error(f"Exception during model lazy loading: {e_load}")
            return [], image_bgr_numpy.shape[0], image_bgr_numpy.shape[1]

    img_h_orig, img_w_orig = image_bgr_numpy.shape[:2]

    # 1. 图像预处理 (letterbox)
    # letterbox 返回: img (处理后的图像), ratio (宽和高的缩放比例元组), pad (宽和高的填充元组 (dw, dh))
    # auto=True: 会自动计算最小的矩形填充以满足stride要求，更灵活
    # auto=False: (通常与rect=True的训练配合) 会严格缩放到new_shape并填充，如果new_shape不是stride倍数会报错
    # 对于推理，auto=True通常更方便。
    img_letterboxed, ratio, pad = letterbox(image_bgr_numpy, new_shape=img_size, stride=MODEL_STRIDE, auto=True)

    # 转换: HWC, BGR -> CHW, RGB
    img_rgb_chw = img_letterboxed.transpose((2, 0, 1))[::-1]
    img_contiguous = np.ascontiguousarray(img_rgb_chw)  # 确保内存连续

    img_tensor = torch.from_numpy(img_contiguous).to(DEVICE).float()
    img_tensor /= 255.0  # 归一化到 0.0 - 1.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度: (CHW) -> (1, CHW)

    # 2. 模型推理
    with torch.no_grad():  # 关闭梯度计算以加速并减少内存使用
        # MODEL(img_tensor)[0] 是因为YOLOv5模型在推理时可能返回一个元组，第一个元素是检测结果
        pred_raw = MODEL(img_tensor)[0]

        # 3. 非极大值抑制 (NMS)
    # pred_raw: (batch_size, num_predictions, num_classes + 5) [cx, cy, w, h, obj_conf, cls_conf1, cls_conf2...]
    # classes=None: 表示不对特定类别进行过滤
    # agnostic_nms=False: 表示NMS是类别相关的 (不同类别的重叠框不会被抑制)
    pred_nms = non_max_suppression(pred_raw, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

    detections_list = []
    # pred_nms 是一个列表，每个元素对应batch中的一张图片的结果
    # 由于我们是单张图片输入 (batch_size=1)，所以只关心 pred_nms[0]
    if pred_nms and len(pred_nms) > 0 and pred_nms[0] is not None and len(pred_nms[0]) > 0:
        detections_on_letterboxed_img = pred_nms[0]  # Tensor (num_detections, 6) -> [x1, y1, x2, y2, conf, cls_id]
        # 这些坐标是相对于 letterbox 处理后的图像 (img_letterboxed) 的

        # 4. 将检测框坐标从 letterbox 尺寸还原到原始图像尺寸
        # 克隆一份以避免原地修改，如果 detections_on_letterboxed_img 后续还有用
        coords_scaled = detections_on_letterboxed_img[:, :4].clone() if \
            isinstance(detections_on_letterboxed_img, torch.Tensor) else \
            detections_on_letterboxed_img[:, :4].copy()

        # letterbox 返回的 ratio 是 (ratio_width, ratio_height)
        # letterbox 返回的 pad 是 (pad_width, pad_height)
        # 如果是等比例缩放，ratio_width == ratio_height
        gain = min(ratio)  # 使用较小的缩放比例，与 scale_coords 内部逻辑类似

        pad_w, pad_h = pad

        coords_scaled[:, [0, 2]] -= pad_w  # 减去宽度方向的填充 (x1, x2)
        coords_scaled[:, [1, 3]] -= pad_h  # 减去高度方向的填充 (y1, y2)
        coords_scaled[:, :4] /= gain  # 除以缩放比例

        # 将坐标限制在原始图像边界内
        coords_scaled[:, [0, 2]] = coords_scaled[:, [0, 2]].clip(0, img_w_orig)  # x1, x2
        coords_scaled[:, [1, 3]] = coords_scaled[:, [1, 3]].clip(0, img_h_orig)  # y1, y2

        # 构建最终的输出列表
        for i in range(detections_on_letterboxed_img.shape[0]):
            xyxy_scaled = coords_scaled[i].round().tolist()  # 四舍五入并转为list
            conf = detections_on_letterboxed_img[i, 4].item()
            cls_id = int(detections_on_letterboxed_img[i, 5].item())

            detections_list.append({
                'class_id': cls_id,
                'class_name': CLASS_NAMES.get(cls_id, f'unknown_cls_{cls_id}'),
                'bbox': xyxy_scaled,  # [xmin, ymin, xmax, ymax] 已经还原到原始图像坐标
                'confidence': conf
            })

    return detections_list, img_h_orig, img_w_orig


# --- (可选) 用于独立测试此模块的示例代码 ---
if __name__ == '__main__':
    LOGGER.info("--- Testing obstacle_detector.py ---")

    # TODO: 【你需要修改这里】创建一个测试图片，或者提供一张实际图片的路径
    # test_image_path = "path/to/your/test_image.jpg"
    # if Path(test_image_path).exists():
    #     dummy_bgr_image = cv2.imread(test_image_path)
    # else:
    #     LOGGER.warning(f"Test image not found at {test_image_path}, using a dummy black image.")
    #     dummy_bgr_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 为确保脚本能独立运行，我们创建一个简单的黑色图像进行测试
    dummy_bgr_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 你可以在上面画一些东西来测试，或者加载一张真实图片
    # cv2.rectangle(dummy_bgr_image, (100, 100), (200, 200), (0,255,0), -1) # 画一个绿色矩形

    LOGGER.info(f"Test image shape: {dummy_bgr_image.shape}")

    # 第一次调用会加载模型
    detections, h, w = detect_obstacles_in_image(dummy_bgr_image, img_size=320, conf_thres=0.1)

    if detections:
        LOGGER.info(f"Detected {len(detections)} objects in the dummy image:")
        for i, det in enumerate(detections):
            LOGGER.info(f"  {i + 1}. Class: {det['class_name']} (ID: {det['class_id']}), "
                        f"Conf: {det['confidence']:.2f}, BBox: {det['bbox']}")
    else:
        LOGGER.info("No objects detected in the dummy image.")

    # (可选) 显示带检测框的图像
    # frame_with_boxes = dummy_bgr_image.copy()
    # for det in detections:
    #     xmin, ymin, xmax, ymax = map(int, det['bbox'])
    #     label = f"{det['class_name']}: {det['confidence']:.2f}"
    #     cv2.rectangle(frame_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #     cv2.putText(frame_with_boxes, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow("Test Detection", frame_with_boxes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    LOGGER.info("--- Obstacle detector test finished ---")