import os
import sys
import zipfile
import subprocess
import requests
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
BIN_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

BINARY_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
ZIP_PATH = BIN_DIR / "realesrgan-binary.zip"

def get_executable_path() -> Path:
    """
    Search recursively inside BIN_DIR to locate the realesrgan-ncnn-vulkan.exe.
    This guarantees it finds the executable even if it is nested inside a subfolder.
    """
    for path in BIN_DIR.rglob("realesrgan-ncnn-vulkan.exe"):
        if path.is_file():
            return path
    return None

def download_realesrgan_binary(status_callback=None) -> Path:
    """
    Downloads and extracts the pre-compiled realesrgan-ncnn-vulkan zip.
    Provides real-time progress callback updates for the Streamlit UI.
    """
    exec_path = get_executable_path()
    if exec_path and exec_path.exists():
        return exec_path

    if status_callback:
        status_callback("Downloading Real-ESRGAN NCNN Vulkan binary (approx. 27MB)...", 0.1)

    # Download ZIP file with streaming to show progress
    response = requests.get(BINARY_URL, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 128  # 128 KB
    downloaded = 0
    
    with open(ZIP_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and status_callback:
                    progress = 0.1 + (downloaded / total_size) * 0.7
                    status_callback(f"Downloading: {downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB...", min(progress, 0.8))

    if status_callback:
        status_callback("Extracting archive...", 0.85)

    # Extract ZIP
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(BIN_DIR)

    # Clean up the downloaded ZIP to keep the workspace clean
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    # Find the executable path
    exec_path = get_executable_path()
    if not exec_path:
        raise FileNotFoundError("Could not find 'realesrgan-ncnn-vulkan.exe' inside the extracted package.")

    # Ensure executable has correct permissions (normally not an issue on Windows, but good practice)
    if os.name != 'nt':
        os.chmod(exec_path, 0o755)

    if status_callback:
        status_callback("Real-ESRGAN binary setup complete!", 1.0)
        
    return exec_path

def run_realesrgan(input_path: str, output_path: str, model_name: str = "realesrgan-x4plus", tile_size: int = 400) -> bool:
    """
    Runs the Real-ESRGAN NCNN Vulkan binary via Python subprocess.
    
    Args:
        input_path: Absolute path to the input image file.
        output_path: Absolute path to save the enhanced upscaled image.
        model_name: Name of the AI model. Options:
                    - 'realesrgan-x4plus' (Default, ultra-detail)
                    - 'realesrgan-x4plus-anime' (Sleek, high line-art quality, removes grain)
                    - 'realesr-animevideov3' (Super fast, clean upscaling)
        tile_size: The tiling size for chunked inference. 
                   Reduces GPU memory usage (VRAM) to prevent out-of-memory errors on 4GB GPUs.
    """
    exec_path = get_executable_path()
    if not exec_path or not exec_path.exists():
        # Try downloading on the fly
        exec_path = download_realesrgan_binary()

    # Check if models directory is present next to the executable
    exec_dir = exec_path.parent
    
    # Construct the command line arguments
    # -i input_path
    # -o output_path
    # -n model_name
    # -t tile_size (to prevent out-of-memory on 4GB VRAM)
    # -g 0 (forces the first available GPU, automatically falls back to CPU if not supported)
    cmd = [
        str(exec_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", model_name,
        "-t", str(tile_size),
        "-g", "0"
    ]
    
    # Execute the subprocess
    # We set stdout and stderr to PIPE to capture warnings/errors
    process = subprocess.run(
        cmd,
        cwd=str(exec_dir),  # Must run in executable's folder so it resolves models path correctly
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0 # Prevent CMD popup on Windows
    )

    if process.returncode != 0:
        error_msg = process.stderr or process.stdout or "Unknown error"
        raise RuntimeError(f"Real-ESRGAN execution failed (code {process.returncode}): {error_msg}")
        
    return True
