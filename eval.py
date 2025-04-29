import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import SuperResolutionModel
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

# --------------------------
# Load image using PIL
# --------------------------
def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size, Image.BICUBIC)
    return img

# --------------------------
# Preprocess image
# --------------------------
def preprocess(img):
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)

# --------------------------
# Postprocess tensor to image
# --------------------------
def postprocess(tensor):
    tensor = tensor.squeeze().clamp(0, 1).detach().cpu()
    return transforms.ToPILImage()(tensor)

# --------------------------
# Compute PSNR and SSIM
# --------------------------
def compute_metrics(sr_img, hr_img):
    sr_np = np.array(sr_img.convert('YCbCr'))[..., 0]
    hr_np = np.array(hr_img.convert('YCbCr'))[..., 0]
    p = psnr(hr_np, sr_np)
    s = ssim(hr_np, sr_np)
    return p, s

# --------------------------
# Main Evaluation Script
# --------------------------
def main():
    # ---- Paths to images ----
    lr_path = 'samples/lr.png'
    hr_path = 'samples/hr.png'
    checkpoint_path = 'checkpoints/ext_lpsr_x8.pth'

    # ---- Load and preprocess ----
    lr_img = load_image(lr_path)
    hr_img = load_image(hr_path)
    lr_tensor = preprocess(lr_img)

    # ---- Load model ----
    model = SuperResolutionModel(
        num_blocks=6,
        in_channels=3,
        growth_channels=64,
        scale_factor=4
    )
    model.eval()

    # ---- Load pretrained weights if available ----
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}. Using randomly initialized weights.")

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_img = postprocess(sr_tensor)

    # ---- Compute metrics ----
    p, s = compute_metrics(sr_img, hr_img)
    print(f'PSNR: {p:.2f} dB')
    print(f'SSIM: {s:.4f}')

    # ---- Display images ----
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr_img)
    axs[0].set_title('Low-Res')
    axs[1].imshow(sr_img)
    axs[1].set_title('Super-Resolved')
    axs[2].imshow(hr_img)
    axs[2].set_title('Ground Truth')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
