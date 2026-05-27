"""Check whether PyTorch can access CUDA."""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch installed: False")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        return

    print("torch installed: True")
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
