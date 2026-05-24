# LamRoNet ⚡🩺
### AI-Powered ECG Super Resolution & Enhancement System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/AI-RealESRGAN-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Vulkan-GPU_Accelerated-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Platform-Windows_|_Linux-black?style=for-the-badge">
</p>

---

# 📌 Overview

LamRoNet is a high-performance AI ECG enhancement platform designed to restore blurry, noisy, or low-resolution electrocardiogram (ECG) scans using deep learning super-resolution technology.

Built with Streamlit and powered by Real-ESRGAN NCNN Vulkan, the system delivers fast GPU-accelerated image restoration without requiring heavy AI frameworks such as PyTorch or OpenCV.

The application is optimized for:

- ECG waveform enhancement
- Medical scan restoration
- Clinical document sharpening
- Noise reduction
- High-resolution ECG visualization

---

# ✨ Core Features

## 🔬 AI ECG Super Resolution

LamRoNet uses advanced deep-learning super-resolution models to:

- Upscale ECG scans up to 4x
- Restore blurry waveform lines
- Recover faded medical grids
- Improve text readability
- Reduce compression artifacts
- Enhance low-quality smartphone captures

---

## ⚡ High-Speed NCNN Vulkan Runtime

Unlike traditional AI systems:

- No CUDA required
- Lightweight inference engine
- Fast startup speed
- GPU acceleration via Vulkan
- CPU fallback supported
- Lower RAM consumption

---

## 🧠 Multiple AI Models

### `realesrgan-x4plus`
Best for:
- Clinical ECG papers
- Text-heavy scans
- Realistic restoration

---

### `realesrgan-x4plus-anime`
Best for:
- Thin waveform lines
- Noise removal
- Smoother ECG curves

---

### `realesr-animevideov3`
Best for:
- CPU-only deployment
- Weak hardware
- Fast cloud processing

---

## 🔍 Pixel-Perfect ECG Inspector

Integrated crop-inspection system allows:

- Zoom-in ECG analysis
- Before/After comparison
- Lead-by-lead inspection
- Pixel-level validation

Perfect for analyzing:
- QRS complexes
- P waves
- T waves
- ST elevation details

---

## ☁️ Smart Cloud Optimization

When running on cloud CPUs:

- Images automatically resized
- Width optimized for speed
- Processing accelerated up to 10x
- Typical runtime: 5–15 seconds

---

## 🎨 Clinical Dashboard UI

Modern medical-style dashboard featuring:

- Glassmorphism design
- Clinical dark mode
- Animated heartbeat effects
- Real-time indicators
- Responsive layout
- Smooth transitions

---

# 🏗️ System Architecture

```text
┌────────────────────┐
│    User Upload     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Streamlit Frontend │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Image Preprocessing│
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Real-ESRGAN Engine │
│   NCNN Vulkan AI   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  AI Super Resolution│
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Enhanced ECG Output│
└────────────────────┘
```

---

# 📂 Project Structure

```text
LamRoNet/
│
├── bin/                     # AI binaries & model files
├── temp/                    # Temporary processing images
├── app.py                   # Main Streamlit application
├── utils.py                 # AI processing utilities
├── styles.css               # Clinical dashboard UI
├── requirements.txt         # Python dependencies
├── packages.txt             # Linux cloud packages
├── assets/                  # Screenshots & demo files
└── README.md
```

---

# ⚙️ System Requirements

## Minimum Requirements

| Component | Requirement |
|---|---|
| Python | 3.9+ |
| RAM | 4GB |
| OS | Windows/Linux/macOS |
| CPU | Dual Core |

---

## Recommended Requirements

| Component | Recommended |
|---|---|
| Python | 3.10 |
| RAM | 8GB+ |
| GPU | Vulkan-supported GPU |
| Storage | SSD |
| VRAM | 4GB+ |

---

# 🖥️ GPU Compatibility

| GPU Brand | Vulkan Support |
|---|---|
| NVIDIA GTX/RTX | ✅ |
| AMD Radeon | ✅ |
| Intel Iris/Xe | ✅ |

---

# 🚀 Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/callme1811/LamRoNet.git
cd LamRoNet
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Launch Application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

# 🐳 Docker Deployment

## Build Docker Image

```bash
docker build -t lamronet .
```

---

## Run Container

```bash
docker run -p 8501:8501 lamronet
```

---

# ☁️ Cloud Deployment

LamRoNet works well on:

- Streamlit Community Cloud
- Hugging Face Spaces
- Railway
- Render
- VPS Linux Servers

---

# 📦 Python Dependencies

```txt
streamlit>=1.30.0
pillow>=10.0.0
requests>=2.31.0
```

---

# 🖼️ Supported Image Formats

| Format | Supported |
|---|---|
| PNG | ✅ |
| JPG | ✅ |
| JPEG | ✅ |

---

# 🧪 Recommended Settings

| Scenario | Recommended Model |
|---|---|
| Blurry ECG scans | realesrgan-x4plus |
| Thin ECG lines | realesrgan-x4plus-anime |
| Weak hardware | realesr-animevideov3 |
| Low VRAM GPU | Lower tile size |

---

# ⚡ Performance Benchmarks

| Device | Model | Processing Time |
|---|---|---|
| RTX 3060 | x4plus | 2–4 sec |
| GTX 1650 | anime | 5–8 sec |
| Intel CPU | animev3 | 10–20 sec |

---

# 🔧 Performance Optimization

## Low VRAM Fix

Reduce Tile Size:

| Tile Size | Usage |
|---|---|
| 400 | High Quality |
| 300 | Balanced |
| 200 | Safe Mode |
| 100 | Ultra Low VRAM |

---

## CPU Optimization

For CPU-only deployment:

- Use `realesr-animevideov3`
- Resize images before upload
- Keep width below 1500px

---

# 📷 Example Workflow

```text
Original ECG Scan
        ↓
Upload to LamRoNet
        ↓
AI Enhancement
        ↓
Zoom & Inspection
        ↓
Download Enhanced ECG
```

---

# 📸 Screenshots

## Dashboard

```text
assets/dashboard.png
```

---

## Before & After

```text
assets/before_after.png
```

---

# 🛠️ Troubleshooting

## Vulkan Runtime Missing

Install latest GPU drivers:

- NVIDIA Drivers
- AMD Adrenalin
- Intel Graphics Drivers

---

## Slow Processing

Possible reasons:

- CPU-only mode
- Large image size
- High tile size
- Low available RAM

---

## Binary Download Failed

Delete:

```text
bin/
```

Restart application to re-download binaries.

---

# 🔒 Privacy & Security

- Images are processed locally
- No external AI API used
- No ECG data uploaded to third-party servers
- Suitable for offline usage

---

# ⚠️ Medical Disclaimer

This project is intended for:

- Educational use
- Research
- Image enhancement purposes

AI-generated outputs may not perfectly preserve original medical information.

Always verify ECG interpretations with qualified healthcare professionals.

---

# 🔮 Future Roadmap

Planned future features:

- Batch ECG enhancement
- DICOM support
- PDF ECG restoration
- OCR medical text extraction
- ECG segmentation AI
- Mobile UI optimization
- Multi-language support
- AI waveform detection

---

# 🤝 Contribution

Contributions are welcome.

## Steps

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push branch
5. Open Pull Request

---

# 📜 License

Currently no license specified.

Recommended licenses:

- MIT License
- Apache 2.0

---

# 🙏 Credits

## Real-ESRGAN

Developed by:
- xinntao

GitHub:
- https://github.com/xinntao/Real-ESRGAN

---

## NCNN Vulkan Runtime

Powered by:
- Tencent NCNN Team

---

# ⭐ Support The Project

If you find this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🧠 Contribute improvements
- 📢 Share with others

---

# 📬 Contact

For questions or collaboration:

- Open an issue
- Submit a pull request
- Contact via GitHub

---

<p align="center">
  <b>LamRoNet — AI ECG Enhancement Powered by Real-ESRGAN ⚡</b>
</p>