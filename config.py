"""
config.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Task-Aware Object Detection
Team Vulcan 629 | DSATM Bengaluru

Central configuration for the entire 10-module system.
Every constant, threshold, model name, and FPGA parameter
lives here. Change here → changes everywhere.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────

# Detection backend
# "yolov8n"   → YOLOv8-nano  (fast, 80 COCO classes)
# "yolow"     → YOLO-World   (open-vocabulary, downloads ~60 MB)
DETECTOR_BACKEND: str = "yolov8n"

# Text / affordance embedding model (384-dim, CPU-friendly)
TEXT_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# CLIP model for visual-text similarity (used in multimodal engine)
# Uses openai/clip-vit-base-patch32 via HuggingFace transformers
CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

# ─────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────

W_SEMANTIC:   float = 0.40   # text affordance similarity
W_VISUAL:     float = 0.25   # CLIP image-text similarity
W_PRIOR:      float = 0.20   # task-prior knowledge
W_PHYSICAL:   float = 0.10   # physical affordance heuristics
W_CONTEXT:    float = 0.05   # scene-context proximity boost

assert abs(W_SEMANTIC + W_VISUAL + W_PRIOR + W_PHYSICAL + W_CONTEXT - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"

# ─────────────────────────────────────────────────────────────────────
# REJECTION GATE
# ─────────────────────────────────────────────────────────────────────

# Hard threshold: if best object final_score < this → no answer
REJECTION_THRESHOLD: float = 0.35

# Adaptive: if top-2 scores are within this margin → uncertainty flag
UNCERTAINTY_MARGIN: float = 0.05

# ─────────────────────────────────────────────────────────────────────
# PRIOR MULTIPLIERS
# ─────────────────────────────────────────────────────────────────────

PRIOR_PREFERRED:  float = 2.2   # ideal object for task
PRIOR_ACCEPTABLE: float = 1.35  # usable substitute
PRIOR_NEUTRAL:    float = 1.00  # unrelated but harmless
PRIOR_PENALISED:  float = 0.05  # wrong or dangerous
PRIOR_NEVER_TOOL: float = 0.02  # person / animal

# ─────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────

YOLO_CONF_THRESHOLD: float = 0.25
YOLO_IOU_THRESHOLD:  float = 0.45
YOLO_DEVICE:         str   = "cpu"
YOLO_IMAGE_SIZE:     int   = 640

# ─────────────────────────────────────────────────────────────────────
# FPGA SYSTOLIC ARRAY PARAMETERS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FPGAConfig:
    """
    Parameters matching the Genesys-2 (Artix-7) FPGA configuration
    used with the VEGA AS1061 RISC-V processor.
    """
    pe_rows:        int   = 8          # systolic array rows
    pe_cols:        int   = 8          # systolic array columns
    clock_mhz:      float = 150.0      # FPGA clock frequency
    int8_bits:      int   = 8          # quantisation precision
    accum_bits:     int   = 32         # accumulator width
    ddr3_bw_gbps:   float = 14.0       # DDR3-2133 bandwidth (GB/s)
    axi_width_bits: int   = 128        # AXI4-Stream data width
    lut_total:      int   = 203_800    # Artix-7 XC7A200T LUTs
    dsp_total:      int   = 740        # Artix-7 DSP48E2 blocks
    bram_total_kb:  int   = 13_140     # Block RAM (KB)

    @property
    def pe_count(self) -> int:
        return self.pe_rows * self.pe_cols

    @property
    def peak_gops(self) -> float:
        return (self.pe_count * 2 * self.clock_mhz * 1e6) / 1e9

FPGA = FPGAConfig()

# ─────────────────────────────────────────────────────────────────────
# VISUALISATION COLOURS  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────

VIZ_GREEN:  tuple = (0,   210,  0)    # selected winner
VIZ_RED:    tuple = (0,   0,   200)   # rejected
VIZ_GREY:   tuple = (130, 130, 130)   # ignored / neutral
VIZ_YELLOW: tuple = (0,   200, 220)   # contextual boost
VIZ_WHITE:  tuple = (255, 255, 255)
VIZ_BLACK:  tuple = (10,  10,  10)

# ─────────────────────────────────────────────────────────────────────
# OBJECTS THAT ARE NEVER TOOLS
# ─────────────────────────────────────────────────────────────────────

NEVER_A_TOOL: set = {
    "person", "dog", "cat", "bird", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
}

# ─────────────────────────────────────────────────────────────────────
# 14 OFFICIAL TASKS  (Nagaraja et al. CVPR 2019)
# ─────────────────────────────────────────────────────────────────────

TASKS: Dict[int, Dict[str, str]] = {
    1:  {"name": "step_on_something",       "query": "What should I use to step on something?"},
    2:  {"name": "sit_comfortably",          "query": "What should I use to sit comfortably?"},
    3:  {"name": "place_flowers",            "query": "What should I use to place flowers?"},
    4:  {"name": "get_potatoes_out_of_fire", "query": "What should I use to get potatoes out of fire?"},
    5:  {"name": "water_plant",              "query": "What should I use to water a plant?"},
    6:  {"name": "get_lemon_out_of_tea",     "query": "What should I use to get lemon out of tea?"},
    7:  {"name": "dig_hole",                 "query": "What should I use to dig a hole?"},
    8:  {"name": "open_bottle_of_beer",      "query": "What should I use to open a bottle of beer?"},
    9:  {"name": "open_parcel",              "query": "What should I use to open a parcel?"},
    10: {"name": "serve_wine",               "query": "What should I use to serve wine?"},
    11: {"name": "pour_sugar",               "query": "What should I use to pour sugar?"},
    12: {"name": "smear_butter",             "query": "What should I use to smear butter?"},
    13: {"name": "extinguish_fire",          "query": "What should I use to extinguish fire?"},
    14: {"name": "pound_carpet",             "query": "What should I use to pound a carpet?"},
}

# ─────────────────────────────────────────────────────────────────────
# TASK PRIORS  (preferred / acceptable / penalised per task)
# ─────────────────────────────────────────────────────────────────────

TASK_PRIORS: Dict[int, Dict[str, List[str]]] = {
    1:  {"preferred": ["skateboard","surfboard","bench"],
         "acceptable": ["suitcase","book"],
         "penalised":  ["wine glass","cup","spoon","fork","knife","bowl","bottle"]},

    2:  {"preferred": ["chair","couch","bench"],
         "acceptable": ["bed"],
         "penalised":  ["wine glass","knife","fork","spoon","bottle","bowl","laptop","cell phone"]},

    3:  {"preferred": ["vase"],
         "acceptable": ["cup","bowl","bottle"],
         "penalised":  ["knife","fork","spoon","chair","couch","laptop","cell phone"]},

    4:  {"preferred": ["fork","knife","spoon"],
         "acceptable": ["scissors","tennis racket"],
         "penalised":  ["wine glass","cup","bowl","bottle","chair","couch","laptop"]},

    5:  {"preferred": ["bottle","cup"],
         "acceptable": ["bowl","sink"],
         "penalised":  ["knife","fork","spoon","chair","couch","laptop","cell phone"]},

    6:  {"preferred": ["fork","spoon"],
         "acceptable": ["knife","cup"],
         "penalised":  ["chair","couch","laptop","cell phone","wine glass","bottle"]},

    7:  {"preferred": ["knife","fork"],
         "acceptable": ["spoon","scissors","baseball bat"],
         "penalised":  ["wine glass","cup","bowl","bottle","chair","couch","laptop"]},

    8:  {"preferred": ["knife","scissors"],
         "acceptable": ["fork","spoon"],
         "penalised":  ["chair","couch","laptop","cell phone","wine glass","bowl"]},

    9:  {"preferred": ["knife","scissors"],
         "acceptable": ["fork"],
         "penalised":  ["wine glass","cup","bowl","bottle","chair","couch","laptop","spoon"]},

    10: {"preferred": ["wine glass"],
         "acceptable": ["cup","bottle"],
         "penalised":  ["knife","fork","spoon","pizza","bowl","chair","couch","laptop"]},

    11: {"preferred": ["bowl","spoon","cup"],
         "acceptable": ["bottle"],
         "penalised":  ["chair","couch","laptop","cell phone","wine glass","knife"]},

    12: {"preferred": ["knife","spoon"],
         "acceptable": ["fork"],
         "penalised":  ["wine glass","cup","bowl","bottle","chair","couch","laptop"]},

    13: {"preferred": ["bottle","cup"],
         "acceptable": ["bowl","fire hydrant"],
         "penalised":  ["knife","fork","spoon","chair","couch","laptop","cell phone"]},

    14: {"preferred": ["baseball bat","umbrella"],
         "acceptable": ["book","bowl","bottle"],
         "penalised":  ["wine glass","cup","knife","fork","spoon","laptop","cell phone"]},
}

# ─────────────────────────────────────────────────────────────────────
# AFFORDANCE DESCRIPTIONS  (rich text for semantic embedding)
# ─────────────────────────────────────────────────────────────────────

COCO_AFFORDANCES: Dict[str, str] = {
    "person":         "a human being, not a tool or object that can be used for physical tasks",
    "bird":           "a living bird animal, not usable as a tool",
    "cat":            "a domestic pet cat, not a tool",
    "dog":            "a domestic pet dog, not a tool",
    "horse":          "a large rideable animal, not a hand tool",
    "sheep":          "a farm animal, not a tool",
    "cow":            "a farm animal for dairy, not a tool",
    "elephant":       "a large wild animal, not a tool",
    "bear":           "a dangerous wild animal, not a tool",
    "zebra":          "a wild striped animal, not a tool",
    "giraffe":        "a tall wild animal, not a tool",
    "bicycle":        "two-wheeled pedal vehicle for riding and transport",
    "car":            "motor vehicle for driving on roads",
    "motorcycle":     "two-wheeled motor vehicle for fast riding",
    "airplane":       "fixed-wing aircraft for air travel",
    "bus":            "large multi-passenger road vehicle",
    "train":          "rail vehicle for mass passenger transport",
    "truck":          "heavy road vehicle for cargo transport",
    "boat":           "watercraft for travel on rivers and sea",
    "traffic light":  "road intersection signal controlling vehicle flow",
    "fire hydrant":   "street water supply valve used by firefighters to access pressurised water to extinguish fires",
    "stop sign":      "regulatory road sign requiring vehicles to halt",
    "parking meter":  "coin-operated device for metering paid street parking",
    "bench":          "long rigid outdoor seat for sitting and resting comfortably, can be stepped on",
    "backpack":       "wearable back bag for carrying personal equipment",
    "umbrella":       "collapsible canopy device for rain protection, solid handle useful for striking and pounding carpets",
    "handbag":        "small carried bag for personal items",
    "tie":            "narrow formal fabric worn around the collar",
    "suitcase":       "rigid travel luggage for carrying clothes, can be stepped on as a platform",
    "frisbee":        "aerodynamic plastic disc for throwing sports",
    "skis":           "long snow-sliding boards strapped to boots",
    "snowboard":      "single wide board for snow slope riding",
    "sports ball":    "round inflated ball for team sports",
    "kite":           "lightweight frame and fabric flown on string in wind",
    "baseball bat":   "long solid cylindrical wooden or aluminium club used for striking, hitting, and pounding objects with force",
    "baseball glove": "leather fielding glove worn in baseball",
    "skateboard":     "flat board on four wheels for riding tricks and stepping on",
    "surfboard":      "long buoyant board for ocean wave surfing",
    "tennis racket":  "handled racket with string mesh for hitting tennis balls",
    "bottle":         "cylindrical container for holding and pouring liquids such as water, wine, or beverages; used to water plants or pour on fire",
    "wine glass":     "elegant stemmed crystal glass vessel specifically designed for serving and drinking wine at dinner tables",
    "cup":            "open-topped cylindrical container for drinking hot tea or cold beverages, can hold small items like lemon slices",
    "fork":           "pronged metal dining utensil for spearing and lifting food from plates and retrieving solid items like lemon from liquids",
    "knife":          "sharp bladed metal utensil for cutting food, slicing bread, opening sealed packages and parcels",
    "spoon":          "rounded bowl-shaped utensil for scooping, stirring, and transferring loose dry ingredients like sugar and spices",
    "bowl":           "deep rounded open container for holding liquid or dry food, mixing ingredients, and scooping or pouring substances like sugar",
    "banana":         "yellow curved tropical fruit for eating",
    "apple":          "round crunchy fruit for eating",
    "sandwich":       "bread-enclosed filling food for eating",
    "orange":         "citrus fruit for eating and juicing",
    "broccoli":       "green cruciferous vegetable for cooking and eating",
    "carrot":         "orange root vegetable for eating raw or cooked",
    "hot dog":        "cooked sausage in a bun for eating",
    "pizza":          "round baked flatbread with cheese and toppings for eating",
    "donut":          "round fried sugary dough ring for eating",
    "cake":           "baked sweet layered dessert for eating at celebrations",
    "chair":          "four-legged furniture piece with back support specifically designed for sitting comfortably for extended periods",
    "couch":          "large padded upholstered sofa furniture for relaxed sitting and reclining comfortably in living rooms",
    "potted plant":   "living plant housed in a soil-filled pot container requiring regular watering with liquid to survive and grow",
    "bed":            "large flat mattress furniture designed for sleeping and comfortable long-term resting",
    "dining table":   "flat horizontal surface for placing food dishes during meals",
    "toilet":         "sanitary fixture for human waste disposal",
    "tv":             "flat panel display screen for viewing video content and television broadcasts",
    "laptop":         "portable folding computer with keyboard for computing tasks",
    "mouse":          "handheld pointing device for computer cursor control",
    "remote":         "wireless handheld controller for operating televisions from distance",
    "keyboard":       "array of keys for typing text input to computers",
    "cell phone":     "handheld wireless communication device for calling and messaging",
    "microwave":      "electric appliance using microwave radiation to heat food quickly",
    "oven":           "enclosed heated chamber for baking and roasting food",
    "toaster":        "electric appliance with slots for toasting bread slices",
    "sink":           "basin with tap providing flowing water for washing hands and dishes, useful for watering plants",
    "refrigerator":   "large insulated electric appliance for keeping food and drinks cold",
    "book":           "bound paper volume for reading text, can be used as a flat rigid surface to press or smash objects",
    "clock":          "timekeeping instrument displaying hours minutes and seconds",
    "vase":           "tall decorative ceramic or glass container designed specifically for holding cut flowers with water",
    "scissors":       "hinged dual-blade cutting tool for cutting paper fabric packaging tape and opening sealed parcels",
    "teddy bear":     "soft plush stuffed toy animal for children",
    "hair drier":     "handheld electric device blowing heated air to dry hair",
    "toothbrush":     "small handled brush with bristles for cleaning teeth",
}