SLEAP-NN provides a modular, PyTorch-based architecture:

### **Data Pipeline**

- Efficient loading of data from SLEAP label files
- Parallelized data loading using PyTorch's multiprocessing for high throughput
- Caching (memory/ disk) to accelerate repeated data access and minimize I/O bottlenecks

### **Model System**

- **Pluggable Design**: Easy to add new backbones/ head modules
- **Backbone Networks**: UNet, ConvNeXt, Swin Transformer  
    - See [Backbone Architectures](models.md#backbone-architectures) for more details.
- **Model Types**: Single Instance, Top-Down, Bottom-Up, Multi-Class (Supervised ID models) variants

> - **Single Instance**: Direct pose prediction for single animals

> - **Top-Down**: Two-stage (centroid → centered-instance) for multi-animal scenarios

> - **Bottom-Up**: Simultaneous keypoint detection and association using Part Affinity Fields (PAFs).

> - **Supervised ID or Multi-Class**: Pose estimation + ID assignment for multi-instance scenarios

> Detailed descriptions of all supported architectures are provided in the [Model Types Guide](models.md).

### **Training Engine**

- PyTorch Lightning integration with custom callbacks
- In-built multi-GPU and distributed training support
- Experiment tracking with visualizers and WandB

### **Inference Pipeline**

- Optimized inference workflow for different model types
- Integration with SLEAP's labeling interface

### **Tracking System**

- Multi-instance tracking across frames
- Flow-shift based tracker for robust tracking
