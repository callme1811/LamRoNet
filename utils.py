import os
import sys
import zipfile
import shutil
import subprocess
import requests
import tempfile
import getpass
from pathlib import Path

<<<<<<< HEAD
# Base directory (Git repo mount, might be read-only on Streamlit Cloud)
=======
>>>>>>> 08e60fc (force realesrgan cpu mode)
BASE_DIR = Path(__file__).resolve().parent

<<<<<<< HEAD
# Define writable directories
# On Windows, we use local folders next to the script.
# On Linux/Cloud, we use a unique folder inside the system temp directory to prevent user collisions.
if os.name == 'nt':
    WORK_DIR = BASE_DIR
    BIN_DIR = WORK_DIR / "bin"
    TEMP_DIR = WORK_DIR / "temp"
else:
    # Use unique folder based on the container username to prevent permission errors
    username = getpass.getuser()
    WORK_DIR = Path(tempfile.gettempdir()) / f"ecg_{username}"
    BIN_DIR = WORK_DIR / "bin"
    TEMP_DIR = WORK_DIR / "temp"

# Ensure folders exist with multiple fail-safe fallbacks
try:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    try:
        BIN_DIR = BASE_DIR / "bin"
        BIN_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Final fallback to standard temp directory root
        BIN_DIR = Path(tempfile.gettempdir())

try:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    try:
        TEMP_DIR = BASE_DIR / "temp"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Final fallback to standard temp directory root
        TEMP_DIR = Path(tempfile.gettempdir())
=======
BIN_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
>>>>>>> 08e60fc (force realesrgan cpu mode)


def get_executable_name() -> str:
    if os.name == "nt":
        return "realesrgan-ncnn-vulkan.exe"
    return "realesrgan-ncnn-vulkan"


def get_executable_path():
    exec_name = get_executable_name()

    for path in BIN_DIR.rglob(exec_name):
        if path.is_file():
            return path

    return None

<<<<<<< HEAD
def download_realesrgan_binary(status_callback=None) -> Path:
    """
    Downloads and extracts the pre-compiled platform-specific realesrgan-ncnn-vulkan zip.
    Supports Windows, Linux (Ubuntu), and macOS.
    On Linux, it also copies the models directory from the repository mount to the writable folder.
    """
=======

def download_realesrgan_binary(status_callback=None):
>>>>>>> 08e60fc (force realesrgan cpu mode)
    exec_path = get_executable_path()

    if exec_path and exec_path.exists():
<<<<<<< HEAD
        if os.name != 'nt':
            try:
                os.chmod(exec_path, 0o755)
            except Exception:
                pass
=======
        if os.name != "nt":
            os.chmod(exec_path, 0o755)
>>>>>>> 08e60fc (force realesrgan cpu mode)
        return exec_path

    if os.name == "nt":
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
    elif sys.platform == "darwin":
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
    else:
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"

    zip_path = BIN_DIR / "realesrgan-binary.zip"

    if status_callback:
<<<<<<< HEAD
        status_callback(f"Initializing Real-ESRGAN for {sys.platform} (Downloading approx. 25-45MB to writable temp)...", 0.1)

    # Download ZIP with streaming
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 128
=======
        status_callback("Đang tải Real-ESRGAN binary...", 0.1)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
>>>>>>> 08e60fc (force realesrgan cpu mode)
    downloaded = 0
    block_size = 1024 * 128

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0 and status_callback:
                    progress = 0.1 + (downloaded / total_size) * 0.7
                    status_callback(
                        f"Đang tải: {downloaded / (1024 * 1024):.2f}MB / {total_size / (1024 * 1024):.2f}MB",
                        min(progress, 0.8),
                    )

    if status_callback:
<<<<<<< HEAD
        status_callback("Extracting archive into writable environment...", 0.85)

    # Extract ZIP into BIN_DIR
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
=======
        status_callback("Đang giải nén Real-ESRGAN...", 0.85)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
>>>>>>> 08e60fc (force realesrgan cpu mode)
        zip_ref.extractall(BIN_DIR)

    if zip_path.exists():
        zip_path.unlink()

<<<<<<< HEAD
    # Find the executable
=======
>>>>>>> 08e60fc (force realesrgan cpu mode)
    exec_path = get_executable_path()

<<<<<<< HEAD
    # Grant execution permissions on Linux/macOS
    if os.name != 'nt':
        try:
            os.chmod(exec_path, 0o755)
        except Exception as e:
            if status_callback:
                status_callback(f"Warning: chmod failed: {str(e)}", 0.9)

    # Copy models from repository mount to BIN_DIR/models if they exist in repo and are missing in temp
    repo_models_dir = BASE_DIR / "bin" / "models"
    dest_models_dir = exec_path.parent / "models"
    if repo_models_dir.exists() and not dest_models_dir.exists():
        if status_callback:
            status_callback("Copying model weights from repository mount...", 0.95)
        try:
            shutil.copytree(repo_models_dir, dest_models_dir)
        except Exception:
            pass  # Fallback to the downloaded model files in the ZIP if copying fails
=======
    if not exec_path:
        raise FileNotFoundError(
            f"Không tìm thấy file chạy '{get_executable_name()}' sau khi giải nén."
        )

    if os.name != "nt":
        os.chmod(exec_path, 0o755)
>>>>>>> 08e60fc (force realesrgan cpu mode)

    if status_callback:
        status_callback("Real-ESRGAN đã sẵn sàng.", 1.0)

    return exec_path

<<<<<<< HEAD
def compile_vk_spoof() -> Path:
    """
    On Linux, compiles a tiny C shared library that hooks Vulkan physical device properties
    to spoof CPU devices (like lavapipe/llvmpipe) as discrete GPUs.
    This prevents NCNN from skipping them and throwing "invalid gpu device" (exit code 255).
    Uses a headerless implementation to bypass any libvulkan-dev package requirements.
    """
    spoof_c_path = BIN_DIR / "vk_spoof.c"
    spoof_so_path = BIN_DIR / "libvk_spoof.so"
    
    if spoof_so_path.exists():
        return spoof_so_path
        
    c_code = """#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>

void vkGetPhysicalDeviceProperties(void* physicalDevice, uint32_t* pProperties) {
    typedef void (*PFN_vkGetPhysicalDeviceProperties)(void*, uint32_t*);
    PFN_vkGetPhysicalDeviceProperties real_func = (PFN_vkGetPhysicalDeviceProperties)dlsym(RTLD_NEXT, "vkGetPhysicalDeviceProperties");
    if (real_func) {
        real_func(physicalDevice, pProperties);
    }
    // deviceType is at index 4 (offset 16)
    if (pProperties && pProperties[4] == 4) {
        pProperties[4] = 2; // Spoof CPU (4) as Discrete GPU (2)
    }
}

void vkGetPhysicalDeviceProperties2(void* physicalDevice, uint32_t* pProperties) {
    typedef void (*PFN_vkGetPhysicalDeviceProperties2)(void*, uint32_t*);
    PFN_vkGetPhysicalDeviceProperties2 real_func = (PFN_vkGetPhysicalDeviceProperties2)dlsym(RTLD_NEXT, "vkGetPhysicalDeviceProperties2");
    if (real_func) {
        real_func(physicalDevice, pProperties);
    }
    // properties.deviceType is at index 8 (offset 32 on 64-bit systems due to pointer padding)
    if (pProperties && pProperties[8] == 4) {
        pProperties[8] = 2; // Spoof CPU (4) as Discrete GPU (2)
    }
}

void vkGetPhysicalDeviceProperties2KHR(void* physicalDevice, uint32_t* pProperties) {
    typedef void (*PFN_vkGetPhysicalDeviceProperties2KHR)(void*, uint32_t*);
    PFN_vkGetPhysicalDeviceProperties2KHR real_func = (PFN_vkGetPhysicalDeviceProperties2KHR)dlsym(RTLD_NEXT, "vkGetPhysicalDeviceProperties2KHR");
    if (real_func) {
        real_func(physicalDevice, pProperties);
    }
    if (pProperties && pProperties[8] == 4) {
        pProperties[8] = 2; // Spoof CPU (4) as Discrete GPU (2)
    }
}
"""
    try:
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        with open(spoof_c_path, "w", encoding="utf-8") as f:
            f.write(c_code)
            
        # Compile using gcc
        cmd = ["gcc", "-shared", "-fPIC", "-o", str(spoof_so_path), str(spoof_c_path), "-ldl"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"gcc compilation failed (code {res.returncode}): {res.stderr}")
        return spoof_so_path
    except Exception as e:
        # Write compile error to a file in TEMP_DIR so we can debug it
        err_log = TEMP_DIR / "compile_error.log"
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            with open(err_log, "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
        # Raise the error
        raise RuntimeError(f"Vulkan spoofer compilation failed: {str(e)}")

def run_realesrgan(input_path: str, output_path: str, model_name: str = "realesrgan-x4plus", tile_size: int = 400, scale: int = 4) -> bool:
    """
    Runs the Real-ESRGAN NCNN Vulkan binary via Python subprocess.
    Automatically falls back to CPU mode if Vulkan GPU acceleration is unavailable.
    """
=======

def run_realesrgan(
    input_path: str,
    output_path: str,
    model_name: str = "realesrgan-x4plus",
    tile_size: int = 200,
) -> bool:
>>>>>>> 08e60fc (force realesrgan cpu mode)
    exec_path = get_executable_path()

    if not exec_path or not exec_path.exists():
        exec_path = download_realesrgan_binary()

    exec_dir = exec_path.parent
<<<<<<< HEAD
    
    # Environment copy to configure Vulkan settings
    env = os.environ.copy()
    if os.name != 'nt':
        # Compile and load our Vulkan deviceType spoofer hook
        # to prevent NCNN from rejecting lavapipe as an "invalid gpu device" and exiting with 255.
        spoof_lib_path = compile_vk_spoof()
        if spoof_lib_path and spoof_lib_path.exists():
            env["LD_PRELOAD"] = str(spoof_lib_path)
    
    # Check if we are running inside a Streamlit Cloud container (CPU-only virtual Linux)
    # If yes, we directly execute CPU mode (-g -1) to prevent the Vulkan device scan from hanging indefinitely.
    is_streamlit_cloud = (os.name != 'nt' and ("/mount/src/" in str(BASE_DIR) or os.environ.get("STREAMLIT_SHARING_ORGANIZATION")))
    
    if is_streamlit_cloud:
        cmd_cpu = [
            str(exec_path),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", model_name,
            "-s", str(scale),
            "-t", str(tile_size),
            "-g", "-1"  # CPU Mode
        ]
        
        process_cpu = subprocess.run(
            cmd_cpu,
            cwd=str(exec_dir),
            env=env,
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

    # --- Standard Local GPU with CPU fallback ---
    cmd_gpu = [
        str(exec_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", model_name,
        "-s", str(scale),
=======

    cmd = [
        str(exec_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", str(model_name),
>>>>>>> 08e60fc (force realesrgan cpu mode)
        "-t", str(tile_size),
        "-g", "-1",
    ]

    process = subprocess.run(
        cmd,
        cwd=str(exec_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    if process.returncode != 0:
<<<<<<< HEAD
        error_msg = process.stderr or process.stdout or ""
        low_err = error_msg.lower()
        is_gpu_err = any(k in low_err for k in ["gpu", "vulkan", "device", "failed", "driver", "openvk", "vk"])
        
        if is_gpu_err or process.returncode in [255, 139, 127]:
            # Rerun with CPU mode (-g -1)
            cmd_cpu = [
                str(exec_path),
                "-i", str(input_path),
                "-o", str(output_path),
                "-n", model_name,
                "-s", str(scale),
                "-t", str(tile_size),
                "-g", "-1"  # CPU Mode
            ]
            
            process_cpu = subprocess.run(
                cmd_cpu,
                cwd=str(exec_dir),
                env=env,
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
=======
        error_msg = process.stderr or process.stdout or "Unknown error"
        raise RuntimeError(
            f"Real-ESRGAN execution failed on CPU mode "
            f"(code {process.returncode}): {error_msg}"
        )

    if not Path(output_path).exists():
        raise FileNotFoundError("Real-ESRGAN chạy xong nhưng không tạo ra ảnh output.")

    return True
>>>>>>> 08e60fc (force realesrgan cpu mode)
