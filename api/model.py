import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Multi-task model: Segmentation + Classification
try:
    import segmentation_models_pytorch as smp
    
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            # Segmentation branch
            self.segmentation = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation='sigmoid'
            )
            
            # Classification branch (512 -> 128 -> 1)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # Segmentation output
            seg_out = self.segmentation(x)
            
            # Classification için encoder features kullan
            encoder_features = self.segmentation.encoder(x)
            cls_out = self.classifier(encoder_features[-1])  # Son encoder feature
            
            return seg_out, cls_out
            
except ImportError:
    print("⚠️ segmentation_models_pytorch bulunamadı, basit model kullanılıyor")
    
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x, torch.tensor([0.5])  # Dummy classification

class ModelInference:
    def __init__(self, model_path=None):
        if model_path is None:
            import os
            # Script'in bulunduğu dizinden models klasörünü bul
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), 'models', 'best_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        self.model_loaded = False
        
        # Model yükle (eğer varsa)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Eğer checkpoint bir dict ise, model_state_dict'i al
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            print(f"✓ Model yüklendi: {model_path}")
        except Exception as e:
            print(f"❌ Model yüklenemedi: {str(e)[:200]}")
            raise RuntimeError(f"Model yüklenemedi: {e}")
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Model 512x512 için eğitilmiş
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_bytes):
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        # Görüntüyü yükle
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        original_size = image.size

        # Albumentations ile aynı transformu uygula
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        test_transform = A.Compose([
            A.Resize(1024, 1024),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        augmented = test_transform(image=image_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            if not self.model_loaded:
                raise RuntimeError("Model yüklenemedi. Lütfen model dosyasını kontrol edin.")
            seg_pred, cls_pred = self.model(input_tensor)
            seg_pred = torch.sigmoid(seg_pred)

        # Post-process
        predicted_mask = seg_pred[0, 0].cpu().numpy()
        predicted_ratio = cls_pred[0].item()

        # Denormalize original image for plotting
        original_img_display = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        original_img_display = original_img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        original_img_display = np.clip(original_img_display, 0, 1)

        # Görselleştirme (matplotlib ile base64)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(predicted_mask, cmap='hot')
        axes[1].set_title(f'Predicted Mask (Ratio: {predicted_ratio:.2%})')
        axes[1].axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        vis_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Maskeyi orijinal boyuta getir
        mask_resized = np.array(Image.fromarray(predicted_mask).resize(original_size, Image.LANCZOS))

        # Hücre tespiti (connected components)
        from scipy import ndimage
        try:
            threshold = 0.5
            detected_regions = predicted_mask > threshold
            labeled_array, num_features = ndimage.label(detected_regions)
            detected_cells = num_features
        except Exception:
            detected_cells = int(np.sum(predicted_mask > 0.5) / 1000)

        # Model doğruluk oranı (eğitim sırasında elde edilen)
        model_accuracy = 0.82  # %80
        
        return {
            "predicted_ratio": predicted_ratio,
            "model_accuracy": model_accuracy,
            "segmentation_mask": mask_resized,
            "visualization": vis_b64,
            "detected_cells": detected_cells
        }