import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")

    # Get total, free, and used memory
    total_mem = torch.cuda.get_device_properties(device).total_memory
    free_mem = torch.cuda.memory_reserved(device)
    allocated_mem = torch.cuda.memory_allocated(device)

    print(f"总显存量：{total_mem / 1024 ** 2:.2f} MiB")
    print(f"已预留的显存量：{free_mem / 1024 ** 2:.2f} MiB")
    print(f"已分配的显存量：{allocated_mem / 1024 ** 2:.2f} MiB")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = reserved_memory - allocated_memory
    available_memory = total_memory - reserved_memory

    print(f"总显存量：{total_memory / 1024 ** 2:.2f} MiB")
    print(f"已预留的显存量：{reserved_memory / 1024 ** 2:.2f} MiB")
    print(f"已分配的显存量：{allocated_memory / 1024 ** 2:.2f} MiB")
    print(f"当前可用的显存量：{free_memory / 1024 ** 2:.2f} MiB")
    print(f"总可用的显存量：{available_memory / 1024 ** 2:.2f} MiB")
    
    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {device_count}")
else:
    print("没有可用的 GPU")