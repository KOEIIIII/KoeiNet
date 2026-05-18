


"""
Insta360全景视频语义分割项目配置文件
"""

import os
import sys
from pathlib import Path
import torch


ROOT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).parent.parent


INPUT_DIR = os.path.join(ROOT_DIR, 'input')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')


VIDEO_FRAME_SKIP = 10
VIDEO_MAX_FRAMES = 100000
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.insv']


FACE_WIDTH = 512
EQR_WIDTH = 1024
EQR_HEIGHT = 512




SEGMENTATION_MODEL_TYPE = "mask2former"


MASK2FORMER_CONFIG = {
    "model_name": "facebook/mask2former-swin-large-cityscapes-semantic",
    "num_classes": 19,
    "batch_size": 4,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}




PEOPLE_DETECTION_MODEL_TYPE = "yolo11"


YOLOV8_CONFIG = {
    "model_size": "m",
    "model_path": "yolov8m.pt",
    "confidence": 0.3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


YOLO11_CONFIG = {
    "model_size": "m",
    "model_path": "yolo11m.pt",
    "confidence": 0.25,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "iou": 0.7,
    "max_det": 300
}


MODEL_NAME = MASK2FORMER_CONFIG["model_name"]
NUM_CLASSES = MASK2FORMER_CONFIG["num_classes"]
BATCH_SIZE = MASK2FORMER_CONFIG["batch_size"]
DEVICE = MASK2FORMER_CONFIG["device"]


OVERLAY_ALPHA = 0.5
BBOX_THICKNESS = 1
MIN_AREA_RATIO = 0.005


EXCLUDE_EDGE_BBOXES = False
EDGE_MARGIN = 2
CONTENT_MARGIN_RATIO = 0.01


PEOPLE_BBOX_THICKNESS = 1
PEOPLE_TEXT_SIZE = 0.5
PEOPLE_TEXT_THICKNESS = 1


NUM_WORKERS = 4


LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(ROOT_DIR, "insta360_segmentation.log")


CACHE_DIR = os.path.join(ROOT_DIR, ".cache")
USE_CACHE = True


CLASS_MAPPING = {
    0: "Road",
    1: "Sidewalk",
    2: "Building",
    3: "Wall",
    4: "Fence",
    5: "Pole",
    6: "TrafficLight",
    7: "TrafficSign",
    8: "Vegetation",
    9: "Terrain",
    10: "Sky",
    11: "Person",
    12: "Rider",
    13: "Car",
    14: "Truck",
    15: "Bus",
    16: "Train",
    17: "Motorcycle",
    18: "Bicycle"
}


PERFORMANCE_MONITORING = {
    "enable_timing": True,
    "enable_memory_tracking": True,
    "log_inference_time": True,
    "benchmark_mode": False
}


EXPERIMENTAL_FEATURES = {
    "use_mixed_precision": True,
    "enable_model_compilation": False,
    "use_tensorrt": False,
    "enable_onnx_export": False
}


CLASS_MAPPING_EN = {
    0: "Road",
    1: "Sidewalk",
    2: "Building",
    3: "Wall",
    4: "Fence",
    5: "Pole",
    6: "Traffic Light",
    7: "Traffic Sign",
    8: "Vegetation",
    9: "Terrain",
    10: "Sky",
    11: "Person",
    12: "Rider",
    13: "Car",
    14: "Truck",
    15: "Bus",
    16: "Train",
    17: "Motorcycle",
    18: "Bicycle"
}





ENABLE_SEGMENT_PIPELINE = False
SEGMENT_SECONDS = 5.0
SEGMENT_OVERLAP = 2.5

ENABLE_VISUAL_SEGMENT_SUMMARY = False

ENABLE_GEO_SYNC = False
GEO_SYNC_GPS_CSV = os.path.join(ROOT_DIR, "output_gps.csv")
GEO_SYNC_TIME_OFFSET_SECONDS = 25.0
GEO_SYNC_EXPORT_WGS84 = True
GEO_SYNC_MAX_GAP_WARNING_SEC = 60.0
GEO_SYNC_USE_EXISTING_SEGMENTS = True
GEO_SYNC_SIDECAR_PATH = ""
GEO_SYNC_FILENAME_TZ_OFFSET_HOURS = 8.0
GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES = True
GEO_SYNC_FRAME_STEP = VIDEO_FRAME_SKIP

ENABLE_WEB_SYNC_EXPORT = False
WEB_SYNC_PREFER_WGS84 = False

ENABLE_SOUNDSCAPE = False
ENABLE_FUSION = False
ENABLE_AGENTS = False
ENABLE_DESIGN = False
ENABLE_DELIVERABLE = False


EXPORT_DEBUG_JSON = False



SOUNDSCAPE_TOP_K_EVENTS = 5
SOUNDSCAPE_ENABLE_PANNS = True
PANNS_DIR = os.path.join(ROOT_DIR, "models", "panns")
PANNS_CHECKPOINT_PATH = os.path.join(PANNS_DIR, "Cnn14_mAP=0.431.pth")
PANNS_LABELS_PATH = os.path.join(PANNS_DIR, "class_labels_indices.csv")
PANNS_FORCE_LOCAL_RESOURCES = True
SOUNDSCAPE_PANNS_EXPORT_DIMS = 16
SOUNDSCAPE_STFT_N_FFT = 1024
SOUNDSCAPE_STFT_HOP = 512
SOUNDSCAPE_ROLLOFF_RATIO = 0.85





BUILD_MODEL_FEATURE_TABLE = True
MODEL_EVENT_VOCAB_TOP_N = 30
MODEL_TOPK_EVENT_VOCAB_TOP_N = 20
MODEL_DROP_HIGH_MISSING = True
MODEL_HIGH_MISSING_THRESHOLD = 0.95


VALIDATION_UNIQUE_SEGMENTS = 60
VALIDATION_HIDDEN_DUPLICATES_PER_RATER = 8
VALIDATION_RANDOM_SEED = 20260310





ZHIPU_AGENT_MODEL = os.getenv("ZHIPU_AGENT_MODEL", "glm-5")
ZHIPU_VISION_QA_MODEL = os.getenv("ZHIPU_VISION_QA_MODEL", "")
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))
AGENT_CACHE_ENABLED = os.getenv("AGENT_CACHE_ENABLED", "1").strip() not in {"0", "false", "False"}
AGENT_DISABLE_LLM = os.getenv("AGENT_DISABLE_LLM", "0").strip() in {"1", "true", "True"}





STEP7_SEED = 20260311
STEP7_REG_CV_SPLITS = 5
STEP7_REG_CV_REPEATS = 20
STEP7_CLASS_MIN_COUNT = 5
STEP7_BOOTSTRAP_SAMPLES = 2000
STEP7_BOOTSTRAP_CI_ALPHA = 0.95
STEP7_ENABLE_BOOTSTRAP = True






STEP75_SEED = 20260311
STEP75_REG_CV_SPLITS = 5
STEP75_REG_CV_REPEATS = 20
STEP75_CLASS_MIN_COUNT = 5
STEP75_REUSE_STEP7_SPLITS = True

STEP75_SCREEN_MISSING_THRESHOLD = 0.95
STEP75_SCREEN_VARIANCE_THRESHOLD = 1e-12
STEP75_SCREEN_CORR_THRESHOLD = 0.95
STEP75_SCREEN_TOPK_VISUAL = 20
STEP75_SCREEN_TOPK_AUDIO = 20
STEP75_SCREEN_TOPK_EARLY = 30
STEP75_SCREEN_MIN_MODALITY_EARLY = 8

STEP75_BOOTSTRAP_SAMPLES = 2000
STEP75_BOOTSTRAP_CI_ALPHA = 0.95






STEP8_TOP_N = 0



VIDEO_FRAME_SKIP = 20
GEO_SYNC_FRAME_STEP = VIDEO_FRAME_SKIP
GEO_SYNC_ALIGN_TO_ANALYSIS_FRAMES = True

ENABLE_GIS_EXPORT = False
GIS_EXPORT_PREFER_WGS84 = True

