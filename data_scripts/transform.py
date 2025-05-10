# data_scripts/transform.py
"""
圖像轉換模組：
定義了一組標準的影像預處理轉換流程，包含：
- Resize 成 224x224
- 轉換為 tensor
- 使用 ImageNet 平均與標準差做 normalize
"""

from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
