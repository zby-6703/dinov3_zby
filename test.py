import torch
print(torch.cuda.get_device_name())
print(torch.cuda.mem_get_info())  # 空闲、总显存
print(torch.cuda.memory_allocated() / 1024**3)
print(torch.cuda.memory_reserved() / 1024**3)