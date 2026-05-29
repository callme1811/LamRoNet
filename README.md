# LamRoNet ⚡🩺

## AI-Powered ECG Paper Rectification, Enhancement & Super Resolution Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/AI-RealESRGAN-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Medical-ECG_Processing-purple?style=for-the-badge">
</p>

<p align="center">
  Intelligent ECG image enhancement system for correcting perspective distortion, restoring image quality, and improving waveform visibility using computer vision and AI super-resolution.
</p>

---

# 📌 Overview

LamRoNet is an advanced ECG image processing platform designed to transform low-quality ECG paper photographs into clear, high-resolution, and clinically readable digital images.

Many ECG records are captured using smartphones under non-ideal conditions, causing:

- Perspective distortion
- Motion blur
- Low resolution
- Poor lighting
- Folded paper artifacts
- Compression noise

LamRoNet addresses these issues through a combination of:

- Perspective correction
- ECG paper flattening
- Image denoising
- Adaptive sharpening
- AI-based super resolution
- Interactive ECG inspection

The platform is intended for educational, research, telemedicine, and medical digitization workflows.

---

# 🎯 Objectives

The primary goals of LamRoNet are:

- Improve ECG image readability
- Restore distorted ECG paper photographs
- Enhance waveform visibility
- Support ECG digitization workflows
- Prepare ECG images for downstream AI analysis
- Facilitate remote consultation and telemedicine
- Improve archived ECG record quality

---

# ✨ Core Features

## 📐 ECG Paper Rectification

Transform skewed ECG photographs into a flat top-down view using homography-based perspective correction.

### Benefits

- Removes perspective distortion
- Standardizes ECG layout
- Preserves waveform geometry
- Improves measurement consistency
- Simplifies downstream processing

---

## 🎯 Interactive Corner Selection

Users can precisely select:

- Top Left Corner
- Top Right Corner
- Bottom Right Corner
- Bottom Left Corner

for accurate ECG paper extraction.

Features include:

- Real-time preview
- Adjustable coordinates
- Interactive controls
- Manual correction support

---

## 🔄 Rotation Correction

Before perspective transformation, images can be rotated to ensure proper orientation.

Supported modes:

- Auto rotation
- Manual clockwise rotation
- Manual counter-clockwise rotation

---

## 🧹 Noise Reduction

Advanced filtering techniques reduce:

- Camera sensor noise
- JPEG compression artifacts
- Scanner imperfections
- Uneven illumination
- Background interference

while preserving important ECG waveform structures.

---

## ✨ Adaptive Sharpening

Enhances:

- ECG waveforms
- Medical annotations
- ECG grid lines
- Diagnostic markings
- Printed information

without introducing excessive artifacts.

---

## 🤖 AI Super Resolution

Optional Real-ESRGAN integration provides:

- 4× image upscaling
- Detail restoration
- Enhanced readability
- Better zoom quality
- Improved visualization of thin ECG lines

Especially useful for:

- Old ECG records
- Archived scans
- Low-resolution smartphone captures

---

## 🔍 Before & After Comparison

Visual comparison tools allow users to inspect:

- Original image
- Flattened image
- Enhanced image
- AI-upscaled image

for quality verification.

---

## 💾 Export Enhanced ECG

Processed images can be downloaded for:

- Clinical documentation
- Research datasets
- Educational materials
- Telemedicine systems

---

# 🏗️ Processing Pipeline

```text
ECG Photograph
       │
       ▼
Image Upload
       │
       ▼
Rotation Correction
       │
       ▼
Corner Selection
       │
       ▼
Perspective Transformation
       │
       ▼
ECG Paper Flattening
       │
       ▼
Noise Reduction
       │
       ▼
Adaptive Sharpening
       │
       ▼
(Optional)
AI Super Resolution
       │
       ▼
Enhanced ECG Output
```

---

# 📂 Project Structure

```text
LamRoNet/
│
├── app.py
├── utils.py
├── styles.css
│
├── weights/
│   └── RealESRGAN_x4plus.pth
│
├── bin/
├── temp/
│
├── requirements.txt
├── packages.txt
│
└── README.md
```

---

# ⚙️ Technology Stack

## Frontend

- Streamlit

## Computer Vision

- OpenCV
- NumPy
- Pillow

## Deep Learning

- Real-ESRGAN
- PyTorch
- TorchVision

## Image Processing

- Perspective Transformation
- Homography Estimation
- Image Enhancement
- Denoising
- Sharpening

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/callme1811/LamRoNet.git
cd LamRoNet
```

---

## Create Virtual Environment

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

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python --version
pip list
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

# 🖼 Supported Formats

| Format | Support |
|--------|---------|
| PNG | ✅ |
| JPG | ✅ |
| JPEG | ✅ |

---

# ⚙️ Configuration

## Real-ESRGAN Weights

Place the model weights inside:

```text
weights/
```

Example:

```text
weights/
└── RealESRGAN_x4plus.pth
```

If weights are unavailable, LamRoNet will continue using traditional image enhancement methods.

---

# 💻 System Requirements

## Minimum Requirements

| Component | Requirement |
|----------|-------------|
| Python | 3.9+ |
| RAM | 4 GB |
| CPU | Dual Core |
| Storage | 1 GB |

---

## Recommended Requirements

| Component | Recommendation |
|----------|----------------|
| Python | 3.10+ |
| RAM | 8 GB+ |
| GPU | NVIDIA RTX Series |
| VRAM | 4 GB+ |
| Storage | SSD |

---

# 📊 Typical Use Cases

## 🏥 Clinical Digitization

Convert photographed ECG sheets into clean digital records.

---

## 🌐 Telemedicine

Improve ECG readability before remote consultations.

---

## 🎓 Medical Education

Generate higher-quality ECG examples for teaching and training.

---

## 🤖 AI Research

Preprocess ECG images before:

- Classification
- Segmentation
- Feature extraction
- Deep learning pipelines

---

## 📚 Medical Archives

Restore old ECG records for long-term preservation.

---

# 📈 Performance Benefits

Compared to raw smartphone ECG captures, LamRoNet can:

- Correct geometric distortion
- Improve waveform visibility
- Enhance ECG grid clarity
- Increase image resolution
- Reduce visual noise
- Improve readability during zooming

---

# 🔒 Privacy & Security

LamRoNet is designed with privacy in mind.

### Data Handling

- Images are processed locally
- No external API required
- No cloud upload required
- No patient data transmission
- Suitable for offline environments

---

# 🛠 Troubleshooting

## Application Does Not Start

Check dependencies:

```bash
pip install -r requirements.txt
```

Verify Python version:

```bash
python --version
```

---

## Real-ESRGAN Not Working

Verify this file exists:

```text
weights/RealESRGAN_x4plus.pth
```

If the file is missing, the application will use OpenCV-based enhancement instead.

---

## Slow Processing

Possible causes:

- Large image size
- CPU-only execution
- Limited RAM
- Missing GPU acceleration
- Large AI upscaling workload

---

# 🗺️ Roadmap

Future planned features:

- Automatic ECG paper detection
- Automatic corner localization
- Batch ECG processing
- PDF ECG restoration
- OCR patient information extraction
- DICOM export support
- ECG waveform segmentation
- Deep-learning denoising
- Mobile-friendly interface
- Multi-language support

---

# 🤝 Contributing

Contributions are welcome.

```text
Fork Repository
      ↓
Create Feature Branch
      ↓
Commit Changes
      ↓
Push Branch
      ↓
Open Pull Request
```

---

# 📜 License

MIT License

---

# ⚠️ Medical Disclaimer

LamRoNet is intended for:

- Research
- Education
- Image enhancement

The software does not provide medical diagnosis.

Enhanced ECG images should not be used as the sole basis for clinical decision-making.

Always consult qualified healthcare professionals and refer to original ECG records when performing medical interpretation.

---

# 👨‍💻 Author

**LamRoNet Development Team**

Building intelligent tools for ECG digitization, enhancement, and medical image restoration.

---

<p align="center">
  <b>LamRoNet ⚡ — Transforming ECG Photographs into Clinically Readable Digital Records</b>
</p>