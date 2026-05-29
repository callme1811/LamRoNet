import cv2
import numpy as np
import os

# PyTorch U-Net Model definition
try:
    import torch
    import torch.nn as nn
    
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
            super().__init__()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Downsample
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature
                
            # Upsample
            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )
                self.ups.append(DoubleConv(feature*2, feature))
                
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
        def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)
                
            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]
            
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]
                
                if x.shape != skip_connection.shape:
                    import torchvision.transforms.functional as TF
                    x = TF.resize(x, size=skip_connection.shape[2:])
                    
                concat_x = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_x)
                
            return self.final_conv(x)
except ImportError:
    UNet = None

def segment_ecg_waveform(image, model_path=None, model_type="OpenCV", C_val=10):
    """
    Segments ECG waveforms from paper background using OpenCV adaptive filters, U-Net, or SegFormer.
    image: RGB numpy array
    """
    if model_type == "OpenCV" or not model_path:
        # High-fidelity OpenCV binarization
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian adaptive threshold to segment dark trace lines
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, C_val
        )
        
        # Clean small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cleaned

    elif model_type == "U-Net":
        if UNet is None:
            raise ImportError("PyTorch is not available.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"U-Net weights file not found: {model_path}")
            
        h_orig, w_orig = image.shape[:2]
        
        # Resize to standard size (e.g. 512x512)
        input_size = 512
        img_resized = cv2.resize(image, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Load weights
        model = UNet(in_channels=3, out_channels=1)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
            
        # Resize back
        pred_mask_orig = cv2.resize((pred_mask * 255).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        return pred_mask_orig

    elif model_type == "SegFormer":
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            from PIL import Image
        except ImportError:
            raise ImportError("Transformers library is not available.")
            
        pil_img = Image.fromarray(image)
        processor = SegformerImageProcessor.from_pretrained(model_path)
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        
        inputs = processor(images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=pil_img.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
        return (pred_mask * 255).astype(np.uint8)
        
    return None

def apply_waveform_grid_overlay(wave_mask, grid_style="Clinical Pink Grid"):
    """
    Overlays a binary wave mask onto a clean background (pink grid or pure white paper).
    wave_mask: grayscale numpy array
    """
    h, w = wave_mask.shape
    
    if grid_style == "Clinical Pink Grid":
        # Create pink grid
        grid = np.full((h, w, 3), 255, dtype=np.uint8)
        # 1mm grids (every 10px)
        grid[::10, :, :] = [230, 230, 255]
        grid[:, ::10, :] = [230, 230, 255]
        # 5mm grids (every 50px)
        grid[::50, :, :] = [180, 180, 255]
        grid[:, ::50, :] = [180, 180, 255]
    else:
        # Pure White
        grid = np.full((h, w, 3), 255, dtype=np.uint8)
        
    # Overlay wave mask (255) as black
    grid[wave_mask > 127] = [0, 0, 0]
    return grid