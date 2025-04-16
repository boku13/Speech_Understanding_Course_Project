import torch
import gc
import os

def free_gpu_memory():
    """Free GPU memory by emptying cache and collecting garbage"""
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    # Display current GPU status
    print_gpu_status()

def print_gpu_status():
    """Print current GPU memory status"""
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        print(f"Free Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() - torch.cuda.memory_reserved()) / (1024**3):.2f} GB")
    else:
        print("CUDA not available")

def run_nvidia_smi():
    """Run nvidia-smi command to show GPU status"""
    os.system('nvidia-smi')

if __name__ == "__main__":
    # Show initial status
    print("Initial GPU Status:")
    print_gpu_status()
    print("\nRunning nvidia-smi:")
    run_nvidia_smi()
    
    # Free memory
    print("\nFreeing GPU memory...")
    free_gpu_memory()
    
    # Show status after freeing
    print("\nGPU Status After Freeing Memory:")
    print_gpu_status()
    print("\nRunning nvidia-smi after freeing memory:")
    run_nvidia_smi() 