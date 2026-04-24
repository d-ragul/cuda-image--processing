import cv2
import numpy as np
import torch
import time

# ==============================
# CPU Processing
# ==============================
def cpu_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# ==============================
# GPU Processing (PyTorch)
# ==============================
def gpu_blur(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_tensor = torch.from_numpy(image).float().to(device)
    
    # Simple operation (simulate processing)
    result = img_tensor / 255.0
    
    return result.cpu().numpy()

# ==============================
# Benchmark Function
# ==============================
def benchmark(image):
    # CPU
    start = time.time()
    cpu_result = cpu_blur(image)
    cpu_time = time.time() - start

    # GPU
    start = time.time()
    gpu_result = gpu_blur(image)
    gpu_time = time.time() - start

    print(f"CPU Time: {cpu_time:.5f} sec")
    print(f"GPU Time: {gpu_time:.5f} sec")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    image = cv2.imread("test.jpg")

    if image is None:
        print("Image not found!")
    else:
        benchmark(image)
