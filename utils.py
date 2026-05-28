import os
import sys
import zipfile
import subprocess
import requests
import platform
import cv2
import numpy as np
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
BIN_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Multi-platform support
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    BINARY_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
else:
    # Linux (Streamlit Cloud is Ubuntu/Debian based)
    BINARY_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"

ZIP_PATH = BIN_DIR / "realesrgan-binary.zip"

def get_executable_path() -> Path:
    """
    Search recursively inside BIN_DIR to locate the realesrgan-ncnn-vulkan executable.
    This guarantees it finds the executable even if it is nested inside a subfolder.
    """
    target = "realesrgan-ncnn-vulkan.exe" if IS_WINDOWS else "realesrgan-ncnn-vulkan"
    for path in BIN_DIR.rglob(target):
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
        status_callback(f"Downloading Real-ESRGAN NCNN Vulkan binary for {platform.system()} (approx. 27MB)...", 0.1)

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
        raise FileNotFoundError(f"Could not find binary in the extracted package.")

    # Ensure executable has correct execution permissions on Linux/macOS
    if not IS_WINDOWS:
        os.chmod(exec_path, 0o755)

    if status_callback:
        status_callback("Real-ESRGAN binary setup complete!", 1.0)
        
    return exec_path

def preprocess_image(input_path: str, output_path: str, denoise_strength: float = 0.0, clahe_clip: float = 2.0, clahe_grid: int = 8, sharpen_weight: float = 0.0) -> None:
    """
    Applies high-quality pre-processing filters for ECG/clinical scans.
    - Denoise: Bilateral filter to smooth paper noise while keeping wave edges sharp.
    - CLAHE: Local contrast enhancement on Lightness channel of LAB color space.
    - Sharpen: Unsharp mask to make wave lines extremely clear before AI super-resolution.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read input image for preprocessing: {input_path}")
        
    # 1. Bilateral Denoising (if enabled)
    if denoise_strength > 0:
        # denoise_strength is typically 1.0 to 10.0
        d = int(denoise_strength * 2) | 1  # make sure it is odd
        d = max(3, min(d, 15))
        img = cv2.bilateralFilter(img, d, denoise_strength * 10.0, denoise_strength * 10.0)
        
    # 2. CLAHE (Local Contrast Limited Adaptive Histogram Equalization)
    if clahe_clip > 0:
        # Convert BGR to LAB color space to process Lightness channel (preserving colors)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    # 3. Unsharp Masking (if enabled)
    if sharpen_weight > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.0 + sharpen_weight, blurred, -sharpen_weight, 0)
        
    cv2.imwrite(output_path, img)

def postprocess_image(input_path: str, output_path: str, mode: str = "Original Enhanced", sharpen_weight: float = 0.5) -> None:
    """
    Applies high-quality post-processing filters after AI super-resolution.
    - Clinical Monochromatic: Converts to high-contrast grayscale (pure black waves on white paper).
    - Waveform Isolation: Attenuates grid colors to make black ECG waves stand out.
    - Post-Sharpen: Unsharp mask to make the upscaled image pixel-perfect.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read input image for postprocessing: {input_path}")
        
    # 1. Apply Clinical Color Modes
    if mode == "Clinical Monochromatic":
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE on grayscale for aggressive wave pop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        # Convert back to 3 channels
        img = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
        
    elif mode == "Waveform Isolation":
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Waves are typically dark (low V channel). So we can threshold the V channel.
        wave_mask = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)
        
        # Clean up grid by making background whiter
        v_new = np.where(wave_mask == 255, v, cv2.add(v, 40)) # brighten background
        
        # Lower the saturation of red/pink grid in the background
        # Red Hues are near 0-15 and 165-180
        red_mask = ((h < 15) | (h > 165)) & (s > 40)
        s_new = np.where((red_mask) & (wave_mask == 0), cv2.subtract(s, 50), s)
        
        hsv_new = cv2.merge((h, s_new, v_new))
        img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        
    # 2. Post-Sharpening (Unsharp Mask)
    if sharpen_weight > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.0 + sharpen_weight, blurred, -sharpen_weight, 0)
        
    cv2.imwrite(output_path, img)

def run_realesrgan(input_path: str, output_path: str, model_name: str = "realesrgan-x4plus", tile_size: int = 400, scale: int = 4) -> bool:
    """
    Runs the Real-ESRGAN NCNN Vulkan binary via Python subprocess.
    Automatically handles GPU vs CPU execution, and falls back to CPU if Vulkan fails.
    """
    exec_path = get_executable_path()
    if not exec_path or not exec_path.exists():
        exec_path = download_realesrgan_binary()

    exec_dir = exec_path.parent
    
    # Base command arguments
    # We attempt GPU 0 first (force Vulkan)
    cmd = [
        str(exec_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", model_name,
        "-t", str(tile_size),
        "-s", str(scale),
        "-g", "0"
    ]
    
    # Try running with GPU first, fallback to CPU (-g -1) if Vulkan fails
    try:
        process = subprocess.run(
            cmd,
            cwd=str(exec_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        
        if process.returncode != 0:
            raise RuntimeError(f"Vulkan GPU failed (code {process.returncode}): {process.stderr or process.stdout}")
            
    except Exception as e:
        print(f"GPU Mode failed, falling back to CPU mode (-g -1): {str(e)}")
        cmd_cpu = cmd.copy()
        
        # Modify the GPU device argument to -1 (CPU)
        try:
            g_idx = cmd_cpu.index("-g")
            cmd_cpu[g_idx + 1] = "-1"
        except ValueError:
            cmd_cpu.extend(["-g", "-1"])
            
        process = subprocess.run(
            cmd_cpu,
            cwd=str(exec_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        
        if process.returncode != 0:
            error_msg = process.stderr or process.stdout or "Unknown error"
            raise RuntimeError(f"Real-ESRGAN execution failed on CPU as well (code {process.returncode}): {error_msg}")
            
    return True
