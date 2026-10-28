import streamlit as st
import trimesh
import tempfile
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bambu Studio AI Config Generator", layout="wide")

# --- PRINTER DATABASE (Bambu Studio Style) ---
PRINTER_DB = {
    "Bambu Lab": {
        "X1 Carbon": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+X1C",
            "bed_size": "256x256x256 mm",
            "kinematics": "CoreXY",
            "max_speed": "500 mm/s",
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL X1C"
        },
        "P1S": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+P1S",
            "bed_size": "256x256x256 mm",
            "kinematics": "CoreXY",
            "max_speed": "500 mm/s",
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL P1S"
        },
        "A1": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+A1",
            "bed_size": "256x256x256 mm",
            "kinematics": "Bedslinger",
            "max_speed": "500 mm/s",
            "direct_drive": True,
            "base_profile": "0.20mm Standard @BBL A1"
        },
        "A1 Mini": {
            "image": "https://placehold.co/400x300/1e1e1e/00ae42?text=Bambu+Lab+A1+Mini",
            "bed_size": "180x180x180 mm",
            "kinematics": "Bedslinger (Cantilever)",
            "max_speed": "500 mm/s",
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
            "direct_drive": True,
            "base_profile": "0.20mm Standard @Creality K1 Max"
        }
    }
}

FILAMENT_TYPES = [
    "Bambu PLA Basic",
    "Bambu PLA Matte",
    "Bambu PETG Basic / HF",
    "Bambu ABS",
    "Bambu ASA",
    "Bambu TPU 95A",
    "Bambu PAHT-CF (Nylon Carbon Fiber)",
    "Generic PLA",
    "Generic PETG",
    "Generic ABS",
    "Generic TPU",
    "Generic Nylon",
    "Generic ASA",
    "Generic PC",
    "Generic PEEK",
    "Generic PP",
    "Generic HIPS",
    "Generic Wood-Filled PLA",
    "Generic Metal-Filled PLA",
    "Generic Carbon-Filled PLA",
    "Generic Flexible Filament (TPE/TPC)",
    "Generic High-Temperature Filament (PEEK, PEI, PPSU)",
    "Generic Composite Filament (CF, GF, Metal, Wood, etc.)"
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
        
        # Calculate how many faces point downwards (Z normal < -0.5) to estimate overhang need
        face_normals = mesh.face_normals
        downward_faces = sum(face_normals[:, 2] < -0.5)
        overhang_ratio = (downward_faces / len(face_normals)) * 100

        stats = {
            "filename": uploaded_file.name,
            "dimensions_mm": f"{extents[0]:.2f} (X) x {extents[1]:.2f} (Y) x {extents[2]:.2f} (Z)",
            "height_z": extents[2],
            "footprint_xy": extents[0] * extents[1],
            "volume_cm3": f"{volume_cm3:.2f} cm3",
            "surface_area_cm2": f"{surface_area_cm2:.2f} cm2",
            "overhang_score": f"{overhang_ratio:.1f}% of faces angle steeply downward"
        }
    finally:
        os.remove(tmp_path)
        
    return stats

# --- HELPER FUNCTION: CALL GEMINI API FOR BAMBU STUDIO JSON ---
def generate_bambu_json(api_key, gemini_model, printer_brand, printer_model, printer_specs, filament, stl_stats, user_notes):
    # Pass explicit API key if available; otherwise genai.Client picks up GEMINI_API_KEY from environment
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    
    profile_name = f"AI_{printer_model}_{filament.split()[1]}_{stl_stats['filename'].split('.')[0]}"
    
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
       
    4. USER PREFERENCES/NOTES:
       - {user_notes if user_notes else "Standard balanced quality and strength."}

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
    - Include an "_ai_reasoning" key (string) explaining why you chose these settings for this specific STL geometry and filament.
    - Include relevant Bambu Studio process parameters such as:
      "layer_height", "initial_layer_print_height", "wall_loops", "top_shell_layers", "bottom_shell_layers",
      "sparse_infill_density" (e.g., "15%"), "sparse_infill_pattern" (e.g., "gyroid", "grid", "crosshatch"),
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
    
    # Parse JSON to extract reasoning and format cleanly
    parsed_json = json.loads(response.text)
    reasoning = parsed_json.pop("_ai_reasoning", "Optimized for your model geometry and filament selection.")
    clean_json_str = json.dumps(parsed_json, indent=4)
        
    return clean_json_str, reasoning

# --- MAIN UI LAYOUT ---
st.title("Bambu Studio AI Config Generator (.json)")
st.markdown("Upload your `.stl` model, select your printer and filament, and let **Google Gemini** generate a custom `.json` process preset ready to import into **Bambu Studio**.")

# Sidebar for Gemini API Key & Model Selection
with st.sidebar:
    st.header("Gemini Agent Setup")
    env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_key = st.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        help="Loaded automatically from .env if configured (GEMINI_API_KEY / GOOGLE_API_KEY), or enter manually. Get your key from Google AI Studio (aistudio.google.com)"
    )
    if env_api_key:
        st.caption("✅ Loaded API key from `.env`")
    
    gemini_model = st.selectbox(
        "Gemini Model",
        [
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite"
        ],
        help="'gemini-2.5-flash' is fast and ideal for quick configs. 'gemini-2.5-pro' uses deep reasoning for tricky engineering prints."
    )
    
    st.markdown("---")
    st.markdown("### How to Import in Bambu Studio:\n1. Open **Bambu Studio**.\n2. Click **File** (top left) -> **Import** -> **Import Configs...**\n3. Select the downloaded `.json` file.\n4. Find your new profile under the **Process (User Presets)** dropdown.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Select Printer & Filament")
    
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
    
    # Optional User Intent
    user_notes = st.text_input("Print Goal (Optional)", placeholder="e.g., Needs high strength (gyroid + 5 walls), or quiet/slow printing...")

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
            
            # Smart Geometry Warnings
            if stl_stats["height_z"] > 100 and stl_stats["footprint_xy"] < 900:
                st.warning("Warning: Tall and thin object detected. Gemini will enable an Outer Brim and lower print speeds to prevent knocking the print over.")

    st.markdown("---")
    
    # Generate Button
    if st.button("Generate Bambu Studio .JSON Preset", type="primary", use_container_width=True, disabled=(not uploaded_stl)):
        effective_key = api_key.strip() if api_key else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
        if not effective_key:
            st.error("Please enter your Gemini API Key in the sidebar or define GEMINI_API_KEY in your `.env` file.")
        else:
            with st.spinner(f"{gemini_model} is crafting your Bambu Studio JSON preset..."):
                try:
                    json_content, reasoning = generate_bambu_json(
                        effective_key, gemini_model, brand, model, printer_specs, filament, stl_stats, user_notes
                    )
                    st.session_state["json_output"] = json_content
                    st.session_state["ai_reasoning"] = reasoning
                    st.session_state["json_filename"] = f"Bambu_{model}_{filament.split()[1]}_{uploaded_stl.name.split('.')[0]}.json".replace(" ", "_")
                    st.success("Bambu Studio JSON Preset generated successfully.")
                except Exception as e:
                    st.error(f"Error communicating with Gemini API: {e}")

# --- OUTPUT & DOWNLOAD SECTION ---
if "json_output" in st.session_state:
    st.markdown("---")
    st.subheader("3. Your Bambu Studio Process Preset (.json)")
    
    st.info(f"**AI Engineering Reasoning:** {st.session_state['ai_reasoning']}")
    
    st.download_button(
        label="Download Bambu Studio .JSON File",
        data=st.session_state["json_output"],
        file_name=st.session_state["json_filename"],
        mime="application/json",
        type="primary"
    )
    
    with st.expander("Preview .JSON File Contents", expanded=True):
        st.code(st.session_state["json_output"], language="json")
