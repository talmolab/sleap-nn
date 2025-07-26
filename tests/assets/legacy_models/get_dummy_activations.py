"""Generate dummy activations from legacy SLEAP models for testing.

This script loads a legacy SLEAP model and generates activations from
zero inputs, saving them for comparison with PyTorch implementations.
"""

import argparse
import os
import numpy as np
import h5py
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy activations from legacy SLEAP models"
    )
    parser.add_argument("model_path", type=str, help="Path to best_model.h5 file")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=None,
        help="Input shape as: batch height width channels (e.g., 1 384 384 1)",
    )
    args = parser.parse_args()

    # Import tensorflow only when running
    import tensorflow as tf

    # Load the model
    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Get input shape from model if not specified
    if args.input_shape is None:
        # Get input shape from model
        if hasattr(model, "input_shape"):
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                # Multi-input model, use first input
                input_shape = input_shape[0]
        else:
            # For functional/subclassed models, get from input layer
            input_shape = model.inputs[0].shape

        # Convert TensorShape to tuple and handle None batch dimension
        input_shape = tuple(input_shape)
        if input_shape[0] is None:
            # Replace None batch dimension with 1
            input_shape = (1,) + input_shape[1:]
    else:
        input_shape = tuple(args.input_shape)

    print(f"Model input shape: {input_shape}")
    print(f"Model output names: {[output.name for output in model.outputs]}")

    # Create dummy input (zeros with appropriate dtype)
    # Keras models typically expect float32
    dummy_input = np.zeros(input_shape, dtype=np.float32)

    # Get model predictions
    print("Running inference with zero input...")
    outputs = model.predict(dummy_input)

    # Handle single vs multiple outputs
    if not isinstance(outputs, list):
        outputs = [outputs]

    # Save outputs
    output_path = os.path.join(os.path.dirname(args.model_path), "dummy_activations.h5")

    # Prepare metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source_model": os.path.abspath(args.model_path),
        "model_filename": os.path.basename(args.model_path),
        "model_directory": os.path.basename(os.path.dirname(args.model_path)),
        "input_shape": list(input_shape),
        "input_dtype": str(dummy_input.dtype),
        "tensorflow_version": tf.__version__,
        "outputs": [],
    }

    # Add model architecture info if available
    if hasattr(model, "count_params"):
        metadata["total_params"] = model.count_params()

    print(f"Saving activations to: {output_path}")
    with h5py.File(output_path, "w") as f:
        # Save outputs with names based on the layer/tensor names
        for i, (output, output_tensor) in enumerate(zip(outputs, model.outputs)):
            # Extract clean layer name from tensor name
            # e.g., "CentroidConfmapsHead/BiasAdd:0" -> "CentroidConfmapsHead"
            tensor_name = output_tensor.name
            if "/" in tensor_name:
                # Take the first part before any operation
                dataset_name = tensor_name.split("/")[0]
            else:
                # Fallback to tensor name without :0 suffix
                dataset_name = tensor_name.split(":")[0]

            f.create_dataset(
                dataset_name, data=output, compression="gzip", compression_opts=9
            )

            # Add output info to metadata
            metadata["outputs"].append(
                {
                    "dataset_name": dataset_name,
                    "tensor_name": tensor_name,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "min_value": float(np.min(output)),
                    "max_value": float(np.max(output)),
                    "mean_value": float(np.mean(output)),
                    "std_value": float(np.std(output)),
                }
            )

            print(f"  Saved {dataset_name}: shape={output.shape}, dtype={output.dtype}")

        # Save metadata as JSON string in a dataset
        metadata_json = json.dumps(metadata, indent=2)
        f.create_dataset(
            "metadata", data=metadata_json, dtype=h5py.string_dtype(encoding="utf-8")
        )
        print(f"  Saved metadata")

    print("Done!")


if __name__ == "__main__":
    main()
