import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees", torch.cuda.device_count(), "GPUs")
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))