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

def get_executable_name() -> str:
    """Return the platform-specific executable name."""
    return "realesrgan-ncnn-vulkan.exe" if os.name == 'nt' else "realesrgan-ncnn-vulkan"

def get_executable_path() -> Path:
    """
    Search recursively inside BIN_DIR to locate the platform-specific executable.
    """
    exec_name = get_executable_name()
    for path in BIN_DIR.rglob(exec_name):
        if path.is_file():
            return path
    return None

def download_realesrgan_binary(status_callback=None) -> Path:
    """
    Downloads and extracts the pre-compiled platform-specific realesrgan-ncnn-vulkan zip.
    Supports Windows, Linux (Ubuntu), and macOS.
    """
    exec_path = get_executable_path()
    if exec_path and exec_path.exists():
        # Ensure correct executable permissions on Unix-based systems
        if os.name != 'nt':
            os.chmod(exec_path, 0o755)
        return exec_path

    # Determine platform-specific binary ZIP URL
    if os.name == 'nt':
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
    elif sys.platform == "darwin":
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
    else:
        # Linux (Ubuntu container on Streamlit Cloud)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"

    zip_path = BIN_DIR / "realesrgan-binary.zip"

    if status_callback:
        status_callback(f"Downloading Real-ESRGAN NCNN Vulkan for {sys.platform} (approx. 25-45MB)...", 0.1)

    # Download with streaming
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 128  # 128 KB
    downloaded = 0
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and status_callback:
                    progress = 0.1 + (downloaded / total_size) * 0.7
                    status_callback(f"Downloading: {downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB...", min(progress, 0.8))

    if status_callback:
        status_callback("Extracting platform-specific archive...", 0.85)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BIN_DIR)

    # Clean up the zip file
    if zip_path.exists():
        zip_path.unlink()

    # Locate executable
    exec_path = get_executable_path()
    if not exec_path:
        raise FileNotFoundError(f"Could not find '{get_executable_name()}' inside the extracted package.")

    # Grant execution permissions on Linux/macOS
    if os.name != 'nt':
        os.chmod(exec_path, 0o755)

    if status_callback:
        status_callback("Real-ESRGAN setup completed successfully!", 1.0)
        
    return exec_path

def run_realesrgan(input_path: str, output_path: str, model_name: str = "realesrgan-x4plus", tile_size: int = 400) -> bool:
    """
    Runs the Real-ESRGAN NCNN Vulkan binary via Python subprocess.
    Automatically falls back to CPU mode if Vulkan GPU acceleration is unavailable (e.g. on Streamlit Cloud).
    """
    exec_path = get_executable_path()
    if not exec_path or not exec_path.exists():
        exec_path = download_realesrgan_binary()

    exec_dir = exec_path.parent
    
    # Try GPU first
    cmd_gpu = [
        str(exec_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", model_name,
        "-t", str(tile_size),
        "-g", "0"
    ]
    
    # Run GPU command
    process = subprocess.run(
        cmd_gpu,
        cwd=str(exec_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )

    if process.returncode != 0:
        error_msg = process.stderr or process.stdout or ""
        
        # Check if the failure is due to lack of GPU/Vulkan drivers (very typical on cloud containers)
        # We automatically fall back to CPU mode (-g -1)
        low_err = error_msg.lower()
        is_gpu_err = any(k in low_err for k in ["gpu", "vulkan", "device", "failed", "driver", "openvk", "vk"])
        
        if is_gpu_err or process.returncode == 255 or process.returncode == 139:
            # Rerun with CPU mode (-g -1)
            cmd_cpu = [
                str(exec_path),
                "-i", str(input_path),
                "-o", str(output_path),
                "-n", model_name,
                "-t", str(tile_size),
                "-g", "-1"  # CPU Mode
            ]
            
            process_cpu = subprocess.run(
                cmd_cpu,
                cwd=str(exec_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if process_cpu.returncode == 0:
                return True
            else:
                cpu_error = process_cpu.stderr or process_cpu.stdout or "Unknown CPU error"
                raise RuntimeError(f"Real-ESRGAN execution failed on CPU mode (code {process_cpu.returncode}): {cpu_error}")
        else:
            raise RuntimeError(f"Real-ESRGAN execution failed (code {process.returncode}): {error_msg}")
            
    return True
