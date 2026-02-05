"""
Export model to ONNX format for deployment.

ONNX (Open Neural Network Exchange) is an intermediate format for deploying models
across different platforms and frameworks.
"""

import torch
import torch.onnx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model
from omegaconf import OmegaConf


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 14
):
    """
    Export PyTorch model to ONNX format.
    
    Advanced concept: Model deployment and optimization
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export
    print(f"\nExporting model to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimization
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("✓ Export successful!")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
    # Print model size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Model size: {size_mb:.2f} MB")


def main():
    """Main export function."""
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Get checkpoint path
    checkpoint_path = input("Enter checkpoint path: ")
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    print("\nLoading model...")
    model = create_model(OmegaConf.to_container(cfg.model))
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded")
    
    # Export
    image_size = cfg.data.get('image_size', 224)
    input_shape = (1, 3, image_size, image_size)
    
    output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"
    
    export_to_onnx(model, input_shape, str(output_path))
    
    print("\n✅ Export complete!")
    print(f"\nONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
