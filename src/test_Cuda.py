import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device name:      {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Supported archs:  {torch.cuda.get_arch_list()}")

    try:
        x = torch.randn(4, 4, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("\nReal CUDA matmul succeeded -- GPU is fully usable:")
        print(y)
    except RuntimeError as e:
        print("\nCUDA matmul FAILED even though is_available()==True.")
        print("This means your PyTorch build doesn't include sm_120 kernels.")
        print(f"Error: {e}")
        print("\nFix: reinstall with the cu128 (or newer) index, e.g.:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu128")
else:
    print("\nNo CUDA device detected by PyTorch. Check that you installed "
          "the CUDA build of PyTorch (not the CPU-only wheel).")