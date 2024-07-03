### Config file

This document contains the docstrings for the config file required to pass to the `sleap_nn.ModelTrainer` class to train and run inference on a sleap-nn model.
The config file has three main sections:

- 1. `data_config`: Creating a data pipeline.

- 2. `model_config`: Initialise the sleap-nn backbone and head models.

- 3. `trainer_config`: Hyperparameters required to train the model with Lightning.

***Note***: The structure for `train` in data_config is used for validation set as well, with the key: `val`. Similarly, the structure for `train_data_loader` in trainer_config section is used for `val_data_loader`.

- `data_config`: 
    - `provider`: (str) Provider class to read the input sleap files. Only "LabelsReader" supported for the training pipeline.
    - `pipeline`: (str) Pipeline for training data. One of "TopdownConfmaps", "SingleInstanceConfmaps", "CentroidConfmapsPipeline" or "BottomUp".
    - `train`:
        - `labels_path`: (str) Path to `.slp` files
        - `is_rgb`: (bool) True if the image has 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If input has three channels and this
        is set to False, then we convert the image to grayscale (single-channel)
        image.
        - `max_height`: (int) Maximum height the image should be padded to. If not provided, the
        original image size will be retained. Default: None.
        - `max_width`: (int) Maximum width the image should be padded to. If not provided, the
        original image size will be retained. Default: None.
        - `scale`: (float or List[float]) Factor to resize the image dimensions by, specified as either a float scalar or as a 2-tuple of [scale_x, scale_y]. If a scalar is provided, both dimensions are resized by the same factor.
        - `preprocessing`:
            - `crop_hw`: (List[int]) Crop height and width of each instance (h, w) for centered-instance model. 
            - `augmentation_config`:
                - `random crop`: (Dict[float]) {"random_crop_p": None, "random_crop_hw": None}, where *random_crop_p* is the probability of applying random crop and *random_crop_hw* is the desired output size (out_h, out_w) of the crop. Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
                - `use_augmentations`: (bool) True if the data augmentation should be applied to the data, else False.
                - `augmentation`: 
                    - `intensity`: 
                        - `uniform_noise`: (Tuple[float]) Tuple of uniform noise (min_noise, max_noise). Must satisfy 0. <= min_noise <= max_noise <= 1.
                        - `uniform_noise_p`: (float) Probability of applying random uniform noise. *Default*=0.0
                        - `gaussian_noise_mean`: (float) The mean of the gaussian noise distribution.
                        - `gaussian_noise_std`: (float) The standard deviation of the gaussian noise distribution.
                        - `gaussian_noise_p`: (float) Probability of applying random gaussian noise. *Default*=0.0
                        - `contrast`: (List[float]) The contrast factor to apply. *Default*: (1.0, 1.0).
                        - `contrast_p`: (float) Probability of applying random contrast. *Default*=0.0
                        - `brightness`: (float) The brightness factor to apply. *Default*: (1.0, 1.0).
                        - `brightness_p`: (float) Probability of applying random brightness. *Default*=0.0
                    - `geometric`:
                        - `rotation`: (List[float]) Angles in degrees as a scalar float of the amount of rotation. A random angle in (-rotation, rotation) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation.
                        - `scale`: (float) A scaling factor as a scalar float specifying the amount of scaling. A
                        random factor between (1 - scale, 1 + scale) will be sampled and applied to both images and keypoints. If `None`, no scaling augmentation will be applied.
                        - `translate`: (List[float]) tuple of maximum absolute fraction for horizontal and vertical translations. For example, translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range img_height * b < dy < img_height * b. Will not translate by default.
                        - `affine_p`: (float) Probability of applying random affine transformations. *Default*=0.0
                        - `erase_scale`: (List[float]) Range of proportion of erased area against input image. *Default*: (0.0001, 0.01).
                        - `erase_ratio`: (List[float]) Range of aspect ratio of erased area. *Default*: (1, 1).
                        - `erase_p`: (float) Probability of applying random erase. *Default*=0.0
                        - `mixup_lambda`: (float) min-max value of mixup strength. Default is 0-1. *Default*: `None`.
                        - `mixup_p`: (float) Probability of applying random mixup v2. *Default*=0.0
                        - `input_key`: (str) Can be `image` or `instance`. The input_key `instance` expects the KorniaAugmenter to follow the InstanceCropper else `image` otherwise for default.
    - `val`: (Similar to `train` structure)

- `model_config`: 
    - `init_weight`: (str) model weights initialization method. "default" uses kaiming uniform initialization and "xavier" uses Xavier initialization method.
    - `pre_trained_weights`: (str) Pretrained weights file name supported only for ConvNext and SwinT backbones. For ConvNext, one of ["ConvNeXt_Base_Weights","ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
    - `backbone_config`:
        - `backbone_type`: (str) Backbone architecture for the model to be trained. One of "unet", "convnext" or "swint".
        - `backbone_config`: (for UNet)
            - `in_channels`: (int) Number of input channels. Default is 1.
            - `kernel_size`: (int) Size of the convolutional kernels. Default is 3.
            - `filters`: (int) Base number of filters in the network. Default is 32
            - `filters_rate`: (float) Factor to adjust the number of filters per block. Default is 1.5.
            - `max_stride`: (int) Scalar integer specifying the maximum stride that the image must be
            divisible by.
            - `stem_stride`: (int) If not None, will create additional "down" blocks for initial
            downsampling based on the stride. These will be configured identically to the down blocks below.
            - `middle_block`: (bool) If True, add an additional block at the end of the encoder. default: True
            - `up_interpolate`: (bool) If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. Default: True.
            - `stacks`: (int) Number of upsampling blocks in the decoder. Default is 3.
            - `convs_per_block`: (int) Number of convolutional layers per block. Default is 2.
        - `backbone_config`: (for ConvNext)
            - `arch`: (Default is `Tiny` architecture config. No need to provide if `model_type` is provided)
                - `depths`: (List(int)) Number of layers in each block. Default: [3, 3, 9, 3].
                - `channels`: (List(int)) Number of channels in each block. Default: [96, 192, 384, 768].
            - `model_type`: (str) One of the ConvNext architecture types: ["tiny", "small", "base", "large"]. Default: "tiny". 
            - `stem_patch_kernel`: (int) Size of the convolutional kernels in the stem layer. Default is 4.
            - `stem_patch_stride`: (int) Convolutional stride in the stem layer. Default is 2.
            - `in_channels`: (int) Number of input channels. Default is 1.
            - `kernel_size`: (int) Size of the convolutional kernels. Default is 3.
            - `filters_rate`: (float) Factor to adjust the number of filters per block. Default is 1.5.
            - `convs_per_block`: (int) Number of convolutional layers per block. Default is 2.
            - `up_interpolate`: (bool) If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. Default: True.
        - `backbone_config`: (for SwinT. Default is `Tiny` architecture.)
            - `model_type`: (str) One of the ConvNext architecture types: ["tiny", "small", "base"]. Default: "tiny". 
            - `arch`: Dictionary of embed dimension, depths and number of heads in each layer.
            Default is "Tiny architecture".
            {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}
            - `patch_size`: (List[int]) Patch size for the stem layer of SwinT. Default: [4,4].
            - `stem_patch_stride`: (int) Stride for the patch. Default is 2.
            - `window_size`: (List[int]) Window size. Default: [7,7].
            - `in_channels`: (int) Number of input channels. Default is 1.
            - `kernel_size`: (int) Size of the convolutional kernels. Default is 3.
            - `filters_rate`: (float) Factor to adjust the number of filters per block. Default is 1.5.
            - `convs_per_block`: (int) Number of convolutional layers per block. Default is 2.
            - `up_interpolate`: (bool) If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. Default: True.
    - `head_configs`: (Dict) Dictionary having head configs with keys `confmaps` and `pafs`. For eg, BottomUp model has both `confmaps` and `pafs` whereas Centroid model only has `confmaps` key. All the keys follow the same structure as given below:
        - `confmaps`:
            - `head_type`: (str) Name of the head. Supported values are 'SingleInstanceConfmapsHead', 'CentroidConfmapsHead', 'CenteredInstanceConfmapsHead', 'MultiInstanceConfmapsHead', 'PartAffinityFieldsHead', 'ClassMapsHead', 'ClassVectorsHead', 'OffsetRefinementHead'
            - `head_config`:
                - `part_names`: (List[str]) `None` if nodes from `sio.Labels` file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
                - `edges`: (List[str]) `None` if edges from `sio.Labels` file can be used directly. **Note**: Only for 'PartAffinityFieldsHead'. List of indices `(src, dest)` that form an edge. 
                - `anchor_part`: (int) **Note**: Only for 'CenteredInstanceConfmapsHead'. Index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
                - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
                - `output_stride`: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
                - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models.
        - `pafs`: (same structure as that of `confmaps`.**Note**: This section is only for BottomUp model.)

- `trainer_config`: 
    - `train_data_loader`: (**Note**: Any parameters from [Torch's DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) could be used.)
        - `batch_size`: (int) Number of samples per batch or batch size for training data. *Default* = 1.
        - `shuffle`: (bool) True to have the data reshuffled at every epoch. *Default*: False.
        - `num_workers`: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. *Default*: 0.
    - `val_data_loader`: (Similar to `train_data_loader`)
    - `model_ckpt`: (**Note**: Any parameters from [Lightning's ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html) could be used.)
        - `save_top_k`: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False.
        - `save_last`: (bool) When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. *Default*: None.
    - `device`: (str) Device on which torch.Tensor will be allocated. One of the ("cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "msnpu").
    - `trainer_devices`: (int) Number of devices to train on (int), which devices to train on (list or str), or "auto" to select automatically.
    - `trainer_accelerator`: (str) One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises the machine the model is running on and chooses the appropriate accelerator for the `Trainer` to be connected to.
    - `enable_progress_bar`: (bool) When True, enables printing the logs during training.
    - `steps_per_epoch`: (int) Minimum number of iterations in a single epoch. (Useful if model is trained with very few data points). Refer `limit_train_batches` parameter of Torch `Trainer`. If `None`, the number of iterations depends on the number of samples in the train dataset.
    - `max_epochs`: (int) Maxinum number of epochs to run.
    - `seed`: (int) Seed value for the current experiment.
    - `use_wandb`: (bool) True to enable wandb logging.
    - `save_ckpt`: (bool) True to enable checkpointing. 
    - `save_ckpt_path`: (str) Directory path to save the training config and checkpoint files. *Default*: "./"
    - `wandb`:
        - `entity`: (str) Entity of wandb project.
        - `project`: (str) Project name for the wandb project.
        - `name`: (str) Name of the current run.
        - `api_key`: (str) API key. The API key is masked when saved to config files.
        - `wandb_mode`: (str) "offline" if only local logging is required. Default: "None".
        - `log_params`: (List[str]) List of config parameters to save it in wandb logs. For example, to save learning rate from trainer config section, use "trainer_config.optimizer.lr" (provide the full path to the specific config parameter).
    - `optimizer_name`: (str) Optimizer to be used. One of ["Adam", "AdamW"].
    - `optimizer`
        - `lr`: (float) Learning rate of type float. *Default*: 1e-3
        - `amsgrad`: (bool) Enable AMSGrad with the optimizer. *Default*: False
    - `lr_scheduler`
        - `mode`: (str) One of "min", "max". In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. *Default*: "min".
        - `threshold`: (float) Threshold for measuring the new optimum, to only focus on significant changes. *Default*: 1e-4.
        - `threshold_mode`: (str) One of "rel", "abs". In rel mode, dynamic_threshold = best * ( 1 + threshold ) in max mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. *Default*: "rel".
        - `cooldown`: (int) Number of epochs to wait before resuming normal operation after lr has been reduced. *Default*: 0
        - `patience`: (int) Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the third epoch if the loss still hasn’t improved then. *Default*: 10.
        - `factor`: (float) Factor by which the learning rate will be reduced. new_lr = lr * factor. *Default*: 0.1.
        - `min_lr`: (float or List[float]) A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. *Default*: 0.
    - `early_stopping`
        - `stop_training_on_plateau`: (bool) True if early stopping should be enabled.
        - `min_delta`: (float) Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement.
        - `patience`: (int) Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch. 
