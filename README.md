# Bambu Studio AI Preset Generator

An intelligent, geometry-aware configuration generator that creates optimized **Bambu Studio** slicing process presets (`.json`) using mesh analysis and **Google Gemini**.

---

## Overview

Configuring 3D printer slicing parameters manually requires balancing structural strength, print speeds, overhang capabilities, and filament-specific behaviors. The **Bambu Studio AI Preset Generator** automates this workflow:

1. Evaluates the geometric properties of an uploaded `.stl` file (bounding dimensions, surface area, volume, and downward-facing overhang ratios).
2. Synthesizes geometry metrics with user hardware specifications, filament characteristics, and custom print intentions.
3. Uses Google Gemini LLMs to output fully compliant Bambu Studio process preset configurations ready for instant import.

---

## Key Features

- **Automated STL Geometry Analysis**: Uses `trimesh` to extract volumetric data, aspect ratios, surface area, and downward normal distribution to assess overhang risk and bed adhesion vulnerability.
- **Hardware-Aware Presets**: Built-in specifications for popular 3D printers, including:
  - **Bambu Lab**: X1-Carbon, P1S, A1, A1 Mini
  - **Prusa Research**: MK4
  - **Creality**: K1 Max
- **Material-Specific Profiles**: Supports standard and engineering filaments including PLA Basic/Matte, PETG Basic/HF, ABS, ASA, TPU 95A, and PAHT-CF (Nylon Carbon Fiber).
- **Native Bambu Studio JSON Schema Compliance**: Produces valid process preset `.json` files compatible with Bambu Studio's inheritance structure (`setting_id: GP004`, `inherits` base profiles, string-encoded parameters).
- **Engineering Rationale**: Provides actionable AI reasoning explaining parameter decisions (e.g., support angle thresholds, brim additions for high aspect ratios, wall loop counts, and infill pattern selection).
- **Interactive UI**: Clean, responsive web interface built with **Streamlit**.

---

## System Architecture

```text
  ┌──────────────┐       ┌────────────────────────┐
  │   3D Model   │ ───>  │ Mesh Analysis (Trimesh)│
  │    (.stl)    │       │ - Dimensions & Volume  │
  └──────────────┘       │ - Overhang Estimation  │
                         └───────────┬────────────┘
                                     │
  ┌──────────────┐                   │
  │ Printer Spec │                   ▼
  │      &       │ ────> ┌────────────────────────┐       ┌──────────────────────┐
  │ Filament Type│       │ Google Gemini Client   │ ───>  │ Bambu Studio Preset  │
  └──────────────┘       │ (Prompt Engineering &  │       │       (.json)        │
                         │ Schema Enforcement)    │       └──────────────────────┘
  ┌──────────────┐       └────────────────────────┘
  │ User Intent  │                   ▲
  │ (Goals/Notes)│ ──────────────────┘
  └──────────────┘
```

---

## Getting Started

### Prerequisites

- **Python**: Version 3.10 or higher
- **Google AI Studio API Key**: A valid Gemini API key (obtainable at [Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository or open the project directory:**

   ```bash
   cd automate_config
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure environment variables:**
   Copy `.env.example` to `.env` and set your API key:
   ```bash
   cp .env.example .env
   ```
   Set `GEMINI_API_KEY=your_gemini_api_key` inside `.env`.

---

## Usage

1. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```
2. **Configure API Credentials & Model:**
   - If `GEMINI_API_KEY` is defined in `.env`, it will load automatically. Otherwise, enter your API key in the sidebar.
   - Choose a Gemini model:
     - `gemini-2.5-flash`: Fast response, ideal for rapid iterations.
     - `gemini-2.5-pro`: Advanced reasoning for complex geometry or engineering materials.
3. **Set Parameters:**
   - Choose your printer brand and model.
   - Select your target filament material.
   - (Optional) Provide functional goals in the **Print Goal** field (e.g., _"high structural strength with 5 wall loops and gyroid infill"_).
4. **Upload Model & Generate:**
   - Drag and drop your `.stl` file into the upload zone.
   - Review the extracted geometry and smart aspect-ratio warnings.
   - Click **Generate Bambu Studio .JSON Preset**.
5. **Download Preset:**
   - Inspect the generated parameter payload and AI engineering reasoning.
   - Click **Download Bambu Studio .JSON File**.

---

## Importing Presets into Bambu Studio

1. Open **Bambu Studio**.
2. Navigate to **File** &rarr; **Import** &rarr; **Import Configs...**
3. Select the downloaded `.json` file.
4. Your optimized profile will appear under the **Process** &rarr; **User Presets** dropdown.

---

## Project Structure

```text
automate_config/
├── app.py              # Streamlit application, mesh parsing, and GenAI pipeline
├── requirements.txt    # Project dependencies (Streamlit, Trimesh, Google GenAI, etc.)
└── README.md           # Project documentation
```

---

## Technical Specifications

- **UI Framework**: [Streamlit](https://streamlit.io/)
- **Geometry Processing**: [Trimesh](https://trimesh.org/) & [NumPy](https://numpy.org/)
- **AI Inference**: [Google GenAI SDK](https://github.com/google-gemini/deprecations) (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`)
- **Target Slicers**: [Bambu Studio](https://bambulab.com/en/download/studio) (v1.9.0.0+) and compatible derivatives (such as OrcaSlicer)
