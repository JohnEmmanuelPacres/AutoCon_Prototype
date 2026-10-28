import streamlit as st
import trimesh
import tempfile
import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bambu Studio Config Generator", layout="wide")

# --- PRINTER DATABASE (Bambu Studio Style) ---
PRINTER_DB = {
    "Bambu Lab": {
        "X1 Carbon": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+X1C",
            "bed_size": "256x256x256 mm",
            "kinematics": "CoreXY",
            "max_speed": "500 mm/s",
            "max_speed_num": 500,
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL X1C"
        },
        "P1S": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+P1S",
            "bed_size": "256x256x256 mm",
            "kinematics": "CoreXY",
            "max_speed": "500 mm/s",
            "max_speed_num": 500,
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL P1S"
        },
        "A1": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+A1",
            "bed_size": "256x256x256 mm",
            "kinematics": "Bedslinger",
            "max_speed": "500 mm/s",
            "max_speed_num": 500,
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL A1"
        },
        "A1 Mini": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+A1+Mini",
            "bed_size": "180x180x180 mm",
            "kinematics": "Bedslinger (Cantilever)",
            "max_speed": "500 mm/s",
            "max_speed_num": 500,
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL A1M"
        }
    },
    "Prusa Research": {
        "MK4": {
            "image": "https://placehold.co/400x300/1e1e1e/fa6831?text=Prusa+MK4",
            "bed_size": "250x210x220 mm",
            "kinematics": "Bedslinger",
            "max_speed": "200 mm/s",
            "max_speed_num": 200,
            "direct_drive": True,
            "base_profile": "0.20mm Standard @MK4"
        }
    },
    "Creality": {
        "K1 Max": {
            "image": "https://placehold.co/400x300/1e1e1e/2b7de9?text=Creality+K1+Max",
            "bed_size": "300x300x300 mm",
            "kinematics": "CoreXY",
            "max_speed": "600 mm/s",
            "max_speed_num": 600,
            "direct_drive": True,
            "base_profile": "0.20mm Standard @Creality K1 Max"
        }
    }
}

# Material properties for both AI context and Deterministic Algorithm calculations
FILAMENT_DB = {
    "Bambu PLA Basic": {"speed_mult": 1.0, "warp_risk": "low", "flexible": False},
    "Bambu PLA Matte": {"speed_mult": 0.9, "warp_risk": "low", "flexible": False},
    "Bambu PETG Basic / HF": {"speed_mult": 0.85, "warp_risk": "medium", "flexible": False},
    "Bambu ABS": {"speed_mult": 0.8, "warp_risk": "high", "flexible": False},
    "Bambu ASA": {"speed_mult": 0.8, "warp_risk": "high", "flexible": False},
    "Bambu TPU 95A": {"speed_mult": 0.25, "warp_risk": "low", "flexible": True},
    "Bambu PAHT-CF (Nylon Carbon Fiber)": {"speed_mult": 0.6, "warp_risk": "high", "flexible": False},
    "Generic PLA": {"speed_mult": 0.85, "warp_risk": "low", "flexible": False},
    "Generic PETG": {"speed_mult": 0.70, "warp_risk": "medium", "flexible": False},
    "Generic ABS": {"speed_mult": 0.75, "warp_risk": "high", "flexible": False},
    "Generic TPU": {"speed_mult": 0.20, "warp_risk": "low", "flexible": True},
    "Generic Nylon": {"speed_mult": 0.55, "warp_risk": "high", "flexible": False},
    "Generic ASA": {"speed_mult": 0.75, "warp_risk": "high", "flexible": False},
    "Generic PC": {"speed_mult": 0.60, "warp_risk": "high", "flexible": False},
    "Generic PEEK": {"speed_mult": 0.35, "warp_risk": "high", "flexible": False},
    "Generic PP": {"speed_mult": 0.50, "warp_risk": "high", "flexible": False},
    "Generic HIPS": {"speed_mult": 0.75, "warp_risk": "high", "flexible": False},
    "Generic Wood-Filled PLA": {"speed_mult": 0.60, "warp_risk": "low", "flexible": False},
    "Generic Metal-Filled PLA": {"speed_mult": 0.50, "warp_risk": "low", "flexible": False},
    "Generic Carbon-Filled PLA": {"speed_mult": 0.70, "warp_risk": "low", "flexible": False},
    "Generic Flexible Filament (TPE/TPC)": {"speed_mult": 0.20, "warp_risk": "low", "flexible": True},
    "Generic High-Temperature Filament (PEEK, PEI, PPSU)": {"speed_mult": 0.35, "warp_risk": "high", "flexible": False},
    "Generic Composite Filament (CF, GF, Metal, Wood, etc.)": {"speed_mult": 0.55, "warp_risk": "medium", "flexible": False}
}

FILAMENT_TYPES = list(FILAMENT_DB.keys())

OPTIMIZATION_MODES = [
    "Balanced (Recommended)",
    "Cost-Effective (Minimum Filament)",
    "Time-Saving (Fastest Print)",
    "Robust (Maximum Strength)",
    "High Detail (Visual Quality)",
    "Robust + Cost-Effective (Optimized Strength-to-Weight)"
]

# --- HELPER FUNCTION: ANALYZE STL ---
def analyze_stl(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        mesh = trimesh.load(tmp_path)
        extents = mesh.extents  # [X, Y, Z] dimensions in mm
        volume_cm3 = mesh.volume / 1000.0  # Convert mm^3 to cm^3
        surface_area_cm2 = mesh.area / 100.0
        
        # Face normal overhang count (for AI prompt compatibility)
        face_normals = mesh.face_normals
        downward_faces = np.sum(face_normals[:, 2] < -0.5)
        overhang_face_pct = (downward_faces / len(face_normals)) * 100.0

        # Area-weighted steep overhang calculation (>45 deg, Z normal < -0.707) for deterministic algorithm
        face_areas = mesh.area_faces
        steep_overhang_mask = (face_normals[:, 2] < -0.707) & (face_normals[:, 2] > -0.999)
        overhang_area_ratio = (np.sum(face_areas[steep_overhang_mask]) / mesh.area) * 100.0

        # Slenderness ratio: Height (Z) divided by minimum XY base width
        min_base = max(min(extents[0], extents[1]), 1.0)
        slenderness_ratio = extents[2] / min_base

        stats = {
            "filename": uploaded_file.name,
            "x_mm": extents[0],
            "y_mm": extents[1],
            "z_mm": extents[2],
            "dimensions_mm": f"{extents[0]:.2f} (X) x {extents[1]:.2f} (Y) x {extents[2]:.2f} (Z)",
            "height_z": extents[2],
            "footprint_xy": extents[0] * extents[1],
            "volume_cm3_num": volume_cm3,
            "surface_area_cm2_num": surface_area_cm2,
            "volume_cm3": f"{volume_cm3:.2f} cm3",
            "surface_area_cm2": f"{surface_area_cm2:.2f} cm2",
            "overhang_score": f"{overhang_face_pct:.1f}% of faces angle steeply downward ({overhang_area_ratio:.1f}% by surface area)",
            "overhang_area_ratio": overhang_area_ratio,
            "slenderness_ratio": slenderness_ratio
        }
    finally:
        os.remove(tmp_path)
        
    return stats

# --- ENGINE 1: DETERMINISTIC SLICER OPTIMIZATION ALGORITHM ---
def run_slicer_algorithm(printer_model, printer_specs, filament_name, stl_stats, opt_mode):
    filament_props = FILAMENT_DB.get(filament_name, {"speed_mult": 0.75, "warp_risk": "medium", "flexible": False})
    reasoning_log = []

    # 1. Base Parameters by Optimization Mode
    if opt_mode == "Cost-Effective (Minimum Filament)":
        layer_height = 0.24
        wall_loops = 2
        top_layers = 3
        bottom_layers = 3
        infill_density = 8
        infill_pattern = "supportcubic"
        speed_factor = 1.0
        reasoning_log.append("Cost-Effective mode: Reduced walls to 2 and utilized 8% Support Cubic infill to minimize filament usage while supporting top surfaces.")

    elif opt_mode == "Time-Saving (Fastest Print)":
        layer_height = 0.28
        wall_loops = 2
        top_layers = 3
        bottom_layers = 3
        infill_density = 10
        infill_pattern = "adaptivecubic"
        speed_factor = 1.2
        reasoning_log.append("Time-Saving mode: Increased layer height to 0.28mm and elevated print speeds by 20% with Adaptive Cubic infill.")

    elif opt_mode == "Robust (Maximum Strength)":
        layer_height = 0.20
        wall_loops = 5
        top_layers = 5
        bottom_layers = 5
        infill_density = 35
        infill_pattern = "gyroid"
        speed_factor = 0.85
        reasoning_log.append("Robust mode: Set 5 perimeter walls, 35% isotropic Gyroid infill, and slowed speeds by 15% for stronger inter-layer adhesion.")

    elif opt_mode == "High Detail (Visual Quality)":
        layer_height = 0.12
        wall_loops = 3
        top_layers = 6
        bottom_layers = 5
        infill_density = 15
        infill_pattern = "grid"
        speed_factor = 0.70
        reasoning_log.append("High Detail mode: Reduced layer height to 0.12mm and lowered outer wall speeds to minimize surface artifacts.")

    elif opt_mode == "Robust + Cost-Effective (Optimized Strength-to-Weight)":
        layer_height = 0.20
        wall_loops = 4
        top_layers = 4
        bottom_layers = 4
        infill_density = 15
        infill_pattern = "gyroid"
        speed_factor = 0.95
        reasoning_log.append("Strength-to-Weight mode: Prioritized 4 perimeter walls over dense infill (15% Gyroid) where mechanical bending stresses concentrate.")

    else:  # Balanced (Recommended)
        layer_height = 0.20
        wall_loops = 3
        top_layers = 4
        bottom_layers = 4
        infill_density = 15
        infill_pattern = "gyroid"
        speed_factor = 1.0
        reasoning_log.append("Balanced mode: Standard 0.20mm layer height, 3 walls, and 15% Gyroid infill.")

    # 2. Geometry Heuristics Adjustments
    if stl_stats["z_mm"] < 25.0 and stl_stats["volume_cm3_num"] < 10.0 and layer_height > 0.16:
        layer_height = 0.16
        reasoning_log.append("Small geometry detected (<25mm height): Refined layer height to 0.16mm to preserve fine feature resolution.")

    # Overhang / Support Heuristic
    if stl_stats["overhang_area_ratio"] > 3.5:
        enable_support = "1"
        support_threshold = "30"
        if filament_props["flexible"]:
            support_type = "normal(auto)"
            reasoning_log.append(f"Overhangs detected ({stl_stats['overhang_area_ratio']:.1f}% area): Enabled Normal supports (Tree supports are unstable with flexible filaments).")
        else:
            support_type = "tree(auto)"
            reasoning_log.append(f"Overhangs detected ({stl_stats['overhang_area_ratio']:.1f}% area): Enabled Tree supports to reduce material waste and contact scarring.")
    else:
        enable_support = "0"
        support_type = "normal(auto)"
        support_threshold = "30"
        reasoning_log.append(f"Minimal steep overhangs ({stl_stats['overhang_area_ratio']:.1f}% area): Supports disabled to save print time and filament.")

    # Stability & Bed Adhesion Heuristic (Slenderness & Warping)
    if stl_stats["slenderness_ratio"] > 3.0:
        brim_type = "outer_only"
        brim_width = "8"
        speed_factor *= 0.75
        reasoning_log.append(f"High slenderness ratio ({stl_stats['slenderness_ratio']:.1f}): Tall/thin geometry detected. Enabled 8mm outer brim and reduced speeds by 25%.")
    elif filament_props["warp_risk"] == "high" or (stl_stats["x_mm"] > 150 or stl_stats["y_mm"] > 150):
        brim_type = "outer_only"
        brim_width = "5"
        reasoning_log.append("Elevated thermal warp risk (high-temp material or large footprint): Enabled 5mm outer brim to prevent corner lifting.")
    else:
        brim_type = "no_brim"
        brim_width = "0"

    # 3. Kinematics & Material Speed Calculations
    base_max = printer_specs["max_speed_num"]
    kinematics_mult = 0.80 if "Bedslinger" in printer_specs["kinematics"] and stl_stats["z_mm"] > 120 else 1.0
    if kinematics_mult < 1.0:
        reasoning_log.append("Tall print on a Bedslinger printer: Reduced Y-axis movement speeds by 20% to mitigate bed inertia wobble.")

    effective_mult = filament_props["speed_mult"] * speed_factor * kinematics_mult

    outer_wall_speed = int(min(base_max * 0.40, 200) * effective_mult)
    inner_wall_speed = int(min(base_max * 0.60, 300) * effective_mult)
    infill_speed = int(min(base_max * 0.70, 350) * effective_mult)
    initial_layer_speed = int(50 * filament_props["speed_mult"])
    travel_speed = int(base_max if printer_specs["kinematics"] == "CoreXY" else min(base_max, 300))

    outer_wall_speed = max(outer_wall_speed, 20)
    inner_wall_speed = max(inner_wall_speed, 30)
    infill_speed = max(infill_speed, 35)
    initial_layer_speed = max(initial_layer_speed, 15)

    # 4. Estimate Material Usage
    shell_thickness_mm = wall_loops * 0.42
    shell_vol_cm3 = stl_stats["surface_area_cm2_num"] * (shell_thickness_mm / 10.0)
    interior_vol_cm3 = max(stl_stats["volume_cm3_num"] - shell_vol_cm3, 0)
    infill_vol_cm3 = interior_vol_cm3 * (infill_density / 100.0)
    est_plastic_cm3 = shell_vol_cm3 + infill_vol_cm3
    est_weight_g = est_plastic_cm3 * 1.24

    # 5. Construct Valid Bambu Studio JSON
    safe_filament_tag = filament_name.split()[1] if len(filament_name.split()) > 1 else filament_name
    profile_name = f"Algo_{printer_model}_{opt_mode.split()[0]}_{stl_stats['filename'].split('.')[0]}"

    bambu_json = {
        "type": "process",
        "setting_id": "GP004",
        "name": profile_name,
        "from": "User",
        "is_custom_defined": "0",
        "inherits": printer_specs["base_profile"],
        "version": "1.9.0.0",
        "layer_height": f"{layer_height:.2f}",
        "initial_layer_print_height": "0.20",
        "wall_loops": str(wall_loops),
        "top_shell_layers": str(top_layers),
        "bottom_shell_layers": str(bottom_layers),
        "sparse_infill_density": f"{infill_density}%",
        "sparse_infill_pattern": infill_pattern,
        "enable_support": enable_support,
        "support_type": support_type,
        "support_threshold_angle": support_threshold,
        "brim_type": brim_type,
        "brim_width": brim_width,
        "outer_wall_speed": str(outer_wall_speed),
        "inner_wall_speed": str(inner_wall_speed),
        "sparse_infill_speed": str(infill_speed),
        "initial_layer_speed": str(initial_layer_speed),
        "travel_speed": str(travel_speed)
    }

    estimates = {
        "est_weight_g": est_weight_g,
        "outer_wall_speed": outer_wall_speed,
        "infill_speed": infill_speed,
        "layer_height": layer_height
    }

    return json.dumps(bambu_json, indent=4), " ".join(reasoning_log), reasoning_log, estimates

# --- ENGINE 2: GEMINI AI AGENT FOR BAMBU STUDIO JSON ---
def generate_bambu_json_ai(api_key, gemini_model, printer_brand, printer_model, printer_specs, filament, stl_stats, opt_mode, user_notes, algo_baseline_json):
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    
    safe_filament_tag = filament.split()[1] if len(filament.split()) > 1 else filament
    profile_name = f"AI_{printer_model}_{safe_filament_tag}_{stl_stats['filename'].split('.')[0]}"
    
    prompt = f"""
    Generate a valid Bambu Studio Process Preset configuration in `.json` format based on the following parameters:

    1. PRINTER SPECIFICATIONS:
       - Brand & Model: {printer_brand} {printer_model}
       - Kinematics: {printer_specs['kinematics']}
       - Max Speed Capability: {printer_specs['max_speed']}
       - Base System Profile (`inherits`): "{printer_specs['base_profile']}"

    2. FILAMENT SELECTED:
       - Type: {filament}

    3. 3D MODEL GEOMETRY ANALYSIS (.STL):
       - Dimensions: {stl_stats['dimensions_mm']}
       - Volume: {stl_stats['volume_cm3']}
       - Surface Area: {stl_stats['surface_area_cm2']}
       - Overhang Analysis: {stl_stats['overhang_score']}
       - Slenderness Ratio (Height/Base): {stl_stats['slenderness_ratio']:.2f}

    4. OPTIMIZATION OBJECTIVE & USER NOTES:
       - Selected Optimization Mode: {opt_mode}
       - Custom User Notes: {user_notes if user_notes else "None provided."}

    5. ALGORITHMIC BASELINE REFERENCE (Use or refine as needed):
    {algo_baseline_json}

    BAMBU STUDIO JSON SCHEMA REQUIREMENTS:
    - In Bambu Studio JSON files, almost all numbers and booleans are formatted as STRINGS (e.g., "2", "15%", "0", "1", "200").
    - Include the following mandatory metadata keys:
      "type": "process",
      "setting_id": "GP004",
      "name": "{profile_name}",
      "from": "User",
      "is_custom_defined": "0",
      "inherits": "{printer_specs['base_profile']}",
      "version": "1.9.0.0"
    - Include an "_ai_reasoning" key (string) explaining why you chose these settings for this specific STL geometry, optimization objective, and filament.
    - Include relevant Bambu Studio process parameters such as:
      "layer_height", "initial_layer_print_height", "wall_loops", "top_shell_layers", "bottom_shell_layers",
      "sparse_infill_density" (e.g., "15%"), "sparse_infill_pattern" (e.g., "gyroid", "grid", "crosshatch", "supportcubic", "adaptivecubic"),
      "enable_support" ("0" or "1"), "support_type" (e.g., "tree(auto)" or "normal(auto)"), "support_threshold_angle",
      "brim_type" (e.g., "auto_brim", "outer_only", "no_brim"), "brim_width",
      "outer_wall_speed", "inner_wall_speed", "sparse_infill_speed", "initial_layer_speed", "travel_speed"
    """

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are an expert Bambu Studio 3D Printing Engineer. You output strictly valid JSON adhering to Bambu Studio's process preset schema.",
            response_mime_type="application/json",
            temperature=0.2,
        )
    )
    
    parsed_json = json.loads(response.text)
    reasoning = parsed_json.pop("_ai_reasoning", "Optimized for your model geometry and filament selection.")
    clean_json_str = json.dumps(parsed_json, indent=4)
        
    return clean_json_str, reasoning

# --- MAIN UI LAYOUT ---
st.title("Bambu Studio Config Generator (.json)")
st.markdown("Upload your `.stl` model, select your printer and filament, and generate a custom `.json` process preset using either our **Deterministic Heuristic Algorithm** or **Google Gemini AI Agent**.")

# Sidebar for Engine & Gemini Setup
with st.sidebar:
    st.header("1. Configuration Engine")
    engine_choice = st.radio(
        "Select Generation Method",
        [
            "Deterministic Algorithm (Instant / Offline)",
            "Gemini AI Agent (Custom Reasoning)"
        ],
        help="Deterministic Algorithm uses physics and geometric math rules instantaneously. Gemini AI Agent allows custom natural-language print goals and AI refinement."
    )
    
    st.markdown("---")
    if engine_choice == "Gemini AI Agent (Custom Reasoning)":
        st.header("2. Gemini Agent Setup")
        env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        api_key = st.text_input(
            "Gemini API Key",
            value=env_api_key,
            type="password",
            help="Loaded automatically from .env if configured (GEMINI_API_KEY / GOOGLE_API_KEY), or enter manually."
        )
        if env_api_key:
            st.caption("Loaded API key from .env")
        
        gemini_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite"
            ],
            help="Select your preferred Gemini model."
        )
        st.markdown("---")

    st.markdown("### How to Import in Bambu Studio:\n1. Open **Bambu Studio**.\n2. Click **File** (top left) -> **Import** -> **Import Configs...**\n3. Select the downloaded `.json` file.\n4. Find your new profile under the **Process (User Presets)** dropdown.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Select Printer, Filament & Goal")
    
    # Brand & Printer Selection
    brand = st.selectbox("Printer Brand", list(PRINTER_DB.keys()))
    model = st.selectbox("Printer Model", list(PRINTER_DB[brand].keys()))
    printer_specs = PRINTER_DB[brand][model]
    
    # Bambu Studio-style Printer Card
    with st.container(border=True):
        img_col, spec_col = st.columns([1, 1])
        with img_col:
            st.image(printer_specs["image"], use_container_width=True)
        with spec_col:
            st.markdown(f"**{brand} {model}**")
            st.caption(f"**Bed:** {printer_specs['bed_size']}")
            st.caption(f"**Type:** {printer_specs['kinematics']}")
            st.caption(f"**Max Speed:** {printer_specs['max_speed']}")
            st.caption(f"**Inherits:** `{printer_specs['base_profile']}`")

    # Filament Dropdown
    filament = st.selectbox("Filament Material", FILAMENT_TYPES)
    
    # Optimization Mode Dropdown (Used by both Algorithm and AI)
    opt_mode = st.selectbox("Optimization Objective", OPTIMIZATION_MODES)
    
    # Optional User Intent (Enabled when using Gemini AI)
    user_notes = st.text_input(
        "Custom Print Notes (Used by Gemini AI Mode)",
        placeholder="e.g., Needs watertight walls, quiet printing, or specific support overhang...",
        disabled=(engine_choice.startswith("Deterministic"))
    )

with col2:
    st.subheader("2. Upload 3D Model (.STL)")
    uploaded_stl = st.file_uploader("Drag and drop your STL file here", type=["stl"])
    
    stl_stats = None
    if uploaded_stl:
        with st.spinner("Analyzing 3D geometry..."):
            stl_stats = analyze_stl(uploaded_stl)
            
        with st.container(border=True):
            st.markdown("### Extracted Model Geometry")
            st.write(f"**Bounding Box:** `{stl_stats['dimensions_mm']}`")
            st.write(f"**Volume:** `{stl_stats['volume_cm3']}` | **Surface Area:** `{stl_stats['surface_area_cm2']}`")
            st.write(f"**Overhang Estimate:** `{stl_stats['overhang_score']}`")
            st.write(f"**Slenderness Ratio (Height/Base):** `{stl_stats['slenderness_ratio']:.2f}`")
            
            if stl_stats["slenderness_ratio"] > 3.0:
                st.warning("Warning: Tall and thin object detected. An Outer Brim and reduced print speeds will be applied to prevent knocking the print over.")

    st.markdown("---")
    
    # Generate Button / Logic
    if st.button("Generate Bambu Studio .JSON Preset", type="primary", use_container_width=True, disabled=(not uploaded_stl)):
        # Always calculate algorithmic baseline first (instantaneous)
        algo_json, algo_summary, algo_log, estimates = run_slicer_algorithm(
            model, printer_specs, filament, stl_stats, opt_mode
        )
        safe_filament_tag = filament.split()[1] if len(filament.split()) > 1 else filament

        if engine_choice.startswith("Deterministic"):
            st.session_state["json_output"] = algo_json
            st.session_state["reasoning_text"] = algo_summary
            st.session_state["reasoning_log"] = algo_log
            st.session_state["estimates"] = estimates
            st.session_state["engine_used"] = "Deterministic Heuristic Algorithm"
            st.session_state["json_filename"] = f"Algo_{model}_{opt_mode.split()[0]}_{uploaded_stl.name.split('.')[0]}.json".replace(" ", "_")
            st.success("Deterministic Bambu Studio JSON Preset calculated instantaneously.")
        else:
            effective_key = api_key.strip() if api_key else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
            if not effective_key:
                st.error("Please enter your Gemini API Key in the sidebar or define GEMINI_API_KEY in your .env file.")
            else:
                with st.spinner(f"{gemini_model} is crafting your Bambu Studio JSON preset..."):
                    try:
                        ai_json, ai_reasoning = generate_bambu_json_ai(
                            effective_key, gemini_model, brand, model, printer_specs, filament, stl_stats, opt_mode, user_notes, algo_json
                        )
                        st.session_state["json_output"] = ai_json
                        st.session_state["reasoning_text"] = ai_reasoning
                        st.session_state["reasoning_log"] = []
                        st.session_state["estimates"] = estimates
                        st.session_state["engine_used"] = f"Gemini AI Agent ({gemini_model})"
                        st.session_state["json_filename"] = f"AI_{model}_{safe_filament_tag}_{uploaded_stl.name.split('.')[0]}.json".replace(" ", "_")
                        st.success("AI-Assisted Bambu Studio JSON Preset generated successfully.")
                    except Exception as e:
                        st.error(f"Error communicating with Gemini API: {e}")

# --- OUTPUT & DOWNLOAD SECTION ---
if "json_output" in st.session_state:
    st.markdown("---")
    st.subheader(f"3. Your Bambu Studio Process Preset (.json) — [{st.session_state.get('engine_used', '')}]")
    
    if "estimates" in st.session_state:
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Filament Used (Baseline)", f"~{st.session_state['estimates']['est_weight_g']:.1f} g")
        m2.metric("Baseline Layer Height", f"{st.session_state['estimates']['layer_height']:.2f} mm")
        m3.metric("Baseline Wall / Infill Speed", f"{st.session_state['estimates']['outer_wall_speed']} / {st.session_state['estimates']['infill_speed']} mm/s")

    if st.session_state.get("reasoning_log"):
        with st.container(border=True):
            st.markdown("**Algorithmic Decision Log:**")
            for item in st.session_state["reasoning_log"]:
                st.markdown(f"- {item}")
    else:
        st.info(f"**AI Engineering Reasoning:** {st.session_state['reasoning_text']}")
    
    st.download_button(
        label="Download Bambu Studio .JSON File",
        data=st.session_state["json_output"],
        file_name=st.session_state["json_filename"],
        mime="application/json",
        type="primary",
        use_container_width=True
    )
    
    with st.expander("Preview .JSON File Contents", expanded=True):
        st.code(st.session_state["json_output"], language="json")