import os
import sys
import subprocess
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
GENERATOR_DIR = BASE_DIR / "ecg-image-kit" / "codes" / "ecg-image-generator"
DEFAULT_HEA = GENERATOR_DIR / "SampleData" / "PTB_XL_data" / "00001_lr.hea"
DEFAULT_DAT = GENERATOR_DIR / "SampleData" / "PTB_XL_data" / "00001_lr.dat"

def generate_synthetic_ecg(
    input_file: str = None,
    header_file: str = None,
    output_dir: str = None,
    seed: int = 42,
    resolution: int = 200,
    pad_inches: int = 0,
    print_header: bool = False,
    num_columns: int = -1,
    full_mode: str = "II",
    mask_unplotted_samples: bool = False,
    add_qr_code: bool = False,
    hw_text: bool = False,
    wrinkles: bool = False,
    augment: bool = False,
    lead_bbox: bool = False,
    lead_name_bbox: bool = False,
    standard_grid_color: int = 5,
    rotate: int = 0,
    noise: int = 50,
    crop: float = 0.01,
    temperature: int = 40000,
    store_config: int = 2
) -> dict:
    """
    Invokes the ecg-image-generator script via subprocess and returns a dictionary of output paths.
    """
    if output_dir is None:
        output_dir = str(BASE_DIR / "temp" / "generated")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use default sample data if not provided
    if input_file is None or header_file is None:
        # Copy PTB-XL sample files to output directory first if we are using them,
        # or read them from their original location
        temp_input = Path(output_dir) / "00001_lr.dat"
        temp_header = Path(output_dir) / "00001_lr.hea"
        
        # Copy to temp files
        shutil.copy(str(DEFAULT_DAT), str(temp_input))
        shutil.copy(str(DEFAULT_HEA), str(temp_header))
        
        input_file = str(temp_input)
        header_file = str(temp_header)

    # Resolve executable (python inside virtual environment or default)
    venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    # Build command arguments
    cmd = [
        python_exe,
        "gen_ecg_image_from_data.py",
        "-i", str(input_file),
        "-hea", str(header_file),
        "-o", str(output_dir),
        "-se", str(seed),
        "-st", "0",  # start index (0)
        "-r", str(resolution),
        "--pad_inches", str(pad_inches),
        "--num_columns", str(num_columns),
        "--full_mode", str(full_mode),
        "--standard_grid_color", str(standard_grid_color),
        "--store_config", str(store_config),
        "-rot", str(rotate),
        "-noise", str(noise),
        "-c", str(crop),
        "-t", str(temperature),
    ]

    # Flags
    if print_header:
        cmd.append("-ph")
    if mask_unplotted_samples:
        cmd.append("--mask_unplotted_samples")
    if add_qr_code:
        cmd.append("--add_qr_code")
    if hw_text:
        cmd.append("--hw_text")
    if wrinkles:
        cmd.append("--wrinkles")
    if augment:
        cmd.append("--augment")
    if lead_bbox:
        cmd.append("--lead_bbox")
    if lead_name_bbox:
        cmd.append("--lead_name_bbox")

    # Run command inside ecg-image-generator directory
    try:
        result = subprocess.run(
            cmd,
            cwd=str(GENERATOR_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        stdout = result.stdout
        stderr = result.stderr
        success = True
    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr
        success = False
        raise RuntimeError(f"ECG image generator failed: {stderr}\nSTDOUT: {stdout}")

    # Gather generated files
    files = os.listdir(output_dir)
    generated_pngs = [os.path.join(output_dir, f) for f in files if f.endswith(".png")]
    generated_jsons = [os.path.join(output_dir, f) for f in files if f.endswith(".json")]
    csv_coordinates = os.path.join(output_dir, "Coordinates.csv")
    csv_gridsizes = os.path.join(output_dir, "gridsizes.csv")
    
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "pngs": generated_pngs,
        "jsons": generated_jsons,
        "coordinates_csv": csv_coordinates if os.path.exists(csv_coordinates) else None,
        "gridsizes_csv": csv_gridsizes if os.path.exists(csv_gridsizes) else None,
        "output_dir": output_dir
    }
