## Model Types

#### 🔹 Single Instance

- The model predicts the pose of a single animal per frame.
- Useful for datasets where there is exactly one animal present in each frame.
- Simple and fast—no need for instance detection or tracking.

#### 🔹 Top-Down

- **Stage 1: Centroid detection** – The model first predicts the centroid (center point) of each animal in the frame, providing candidate locations for each instance.
- **Stage 2: Centered instance pose estimation** – For each detected centroid, a second model predicts the pose of the animal within a cropped region centered on that centroid.
- This approach enables accurate pose estimation in crowded scenes by focusing on one animal at a time.
- Particularly effective for datasets with moderate to high animal density, where animals are not heavily overlapping.

#### 🔹 Bottom-Up

- Predicts all body part locations (keypoints) and their pairwise associations (Part Affinity Fields, PAFs) for all animals in the frame simultaneously.
- PAFs encode the direction and strength of connections between body parts, enabling the model to group keypoints into individual animals even when they overlap or are occluded.
- Assembles detected keypoints into individual animal instances by solving a global assignment problem based on the predicted PAFs.
- Effective for challenging scenarios with frequent occlusions, close physical contact, or overlapping animals.

#### 🔹 Top-Down ID model

- **Stage 1: Centroid detection** – The model predicts the centroid of each animal instance (same as standard top-down, without classification).
- **Stage 2: Centered instance pose estimation with classification** – For each detected centroid, a second model predicts the pose of the animal using the image cropped around the centroid and also classifies the instance into predefined classes using supervised learning with ground truth track IDs from the training data.
- **Training Requirement**: Multi-class models require ground truth track IDs during training and are used to assign persistent IDs to animals across frames.

#### 🔹 Bottom-Up ID model 

- Predicts all body part locations (keypoints) and their class labels for all animals simultaneously.
- Directly classifies keypoints and groups them into instances with class assignments.
- Assembles detected keypoints into individual animal instances by solving a global assignment problem, while maintaining class-specific groupings.
- **Training Requirement**: Multi-class models require ground truth track IDs during training and are used to assign persistent IDs to animals across frames.



## Backbone Architectures

SLEAP-NN supports three different backbone architectures for feature extraction, each offering unique advantages for pose estimation tasks.

#### UNet

UNet is based on the original U-Net architecture from Ronneberger et al. (2015), which was made modular and adapted for pose estimation in Pereira et al. (2019). It uses an encoder-decoder structure with skip connections to capture both fine-grained details and high-level features. UNet offers the highest flexibility with configurable depth and filters. It supports stem blocks for initial downsampling, middle blocks for additional processing, and variable convolution layers per block, making it ideal for custom architectures.

#### Swin Transformer (SwinT)

Swin Transformer (SwinT) is based on Liu et al. (2021) "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" and uses a hierarchical vision transformer with shifted windows to efficiently process images at multiple scales while maintaining the benefits of self-attention mechanisms. SwinT is also a large model that divides images into non-overlapping windows and applies self-attention within each window, with shifted window mechanisms allowing information flow between windows. It's available in Tiny, Small, and Base configurations and is particularly effective for capturing complex spatial relationships and global context understanding, though it requires significant computational resources. Pretrained ImageNet weights are available for all Tiny, Small, and Base configurations, enabling transfer learning for improved performance.

#### ConvNeXt

ConvNeXt is based on Liu et al. (2022) "A ConvNet for the 2020s" and modernizes traditional convolutional networks by incorporating design principles from vision transformers while maintaining CNN efficiency. It uses a hierarchical structure with depth-wise convolutions, layer normalization, and modern activation functions. ConvNeXt is a large model that supports ImageNet pre-trained weights for transfer learning and is available in Tiny, Small, Base, and Large configurations, making it ideal for high-performance applications with standard image sizes. Pretrained ImageNet weights are available for all Tiny, Small, Base, and Large configurations, allowing easy initialization for transfer learning.

## How to configure models?

To train or use a specific model type, you must set the corresponding `head_configs` in the `model_config` section of your configuration file. Different model types require different head configurations:

- **Single Instance models** use one head (`single_instance`)
- **Top-Down models** each model (centroid and centered-instance) use one head (`confmaps`)
- **Bottom-Up models** use two heads (`bottomup` with both `confmaps` and `pafs`)
- **ID models** use two heads (e.g., `multi_class_bottomup` with both `confmaps` and `class_maps`/ `class_vectors`)

The choice and configuration of these heads determine the model's outputs and behavior, so it's important to set them carefully according to your task.

For backbone configuration, you can specify the architecture type and parameters in the `backbone_config` section. Each backbone type (UNet, ConvNeXt, SwinT) has its own configuration options that control the network architecture, such as filter counts, depths, and other architectural parameters.

For a detailed explanation of all `model_config`, `head_configs`, and `backbone_config` options—including how to specify multiple heads and the meaning of each parameter—see the [backbone_config section](config.md#backbone-configuration) and [head_configs section](config.md#head-configuration).