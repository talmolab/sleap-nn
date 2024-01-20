##### Config file 

This document contains the docstrings for the config file required to pass to the `sleap_nn.ModelTrainer` class to train a sleap-nn model.
The config file has three main sections:
    1. `data_config` has all the parameters required for creating a data pipeline
    2. `model_config` has the configs to initialise the sleap-nn UNet model
    3. `trainer_config` has the hyperparameters required to train the model with Lightning with optional logging

***Note***: The structure for `train` in data_config is used for validation set with the keys: `val`. Similarly, the structure for `train_data_loader` in trainer_config section  is used for `val_data_loader`.

- `data_config`: 
    - `provider`: provider class to read the input sleap files. (only LabelsReader supported)
    - `pipeline`: Pipeline for Data. One of "TopdownConfmaps", "SingleInstanceConfmaps"
    - `train`:
        - `labels_path`: path to `.slp` files
        - `general`:
            - `keep_keys`: the keys to be retained in the input dictionary throughout the pipeline. Available keys include image, video_idx, frame_idx, instances, centroids, instance, instance_bbox, instance_image, confidence_maps
        - `preprocessing`:
            - `anchor_ind`:   index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
            - `crop_hw`: crop height and width of each instance (h, w)
            - `conf_map_gen`: dictionary in the format {"sigma": 1.5, "output_stride": 2}. *sigma* defines the spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.  *output_stride* defines the stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
            - `augmentation_config`:
                - `random crop`: {"random_crop_p": None, "random_crop_hw": None}, where *random_crop_p* is the probability of applying random crop and *random_crop_hw* is the desired output size (out_h, out_w) of the crop. Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
                - `use_augmentations`: True if the data augmentation should be applied to the data, else False.
                - `augmentation`: 
                    - `intensity`: 
                        - `uniform_noise`: tuple of uniform noise (min_noise, max_noise). Must satisfy 0. <= min_noise <= max_noise <= 1.
                        - `uniform_noise_p`: Probability of applying random uniform noise. *Default*=0.0
                        - `gaussian_noise_mean`: The mean of the gaussian distribution.
                        - `gaussian_noise_std`: The standard deviation of the gaussian distribution.
                        - `gaussian_noise_p`: Probability of applying random gaussian noise. *Default*=0.0
                        - `contrast`: The contrast factor to apply. *Default*: (1.0, 1.0).
                        - `contrast_p`: Probability of applying random contrast. *Default*=0.0
                        - `brightness`: The brightness factor to apply. *Default*: (1.0, 1.0).
                        - `brightness_p`: Probability of applying random brightness. *Default*=0.0
                    - `geometric`:
                        - `rotation`: Angles in degrees as a scalar float of the amount of rotation. A random angle in (-rotation, rotation) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation.
                        - `scale`: A scaling factor as a scalar float specifying the amount of scaling. A
                        random factor between (1 - scale, 1 + scale) will be sampled and applied to both images and keypoints. If `None`, no scaling augmentation will be applied.
                        - `translate`: tuple of maximum absolute fraction for horizontal and vertical translations. For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range img_height * b < dy < img_height * b. Will not translate by default.
                        - `affine_p`: Probability of applying random affine transformations. *Default*=0.0
                        - `erase_scale`: Range of proportion of erased area against input image. *Default*: (0.0001, 0.01).
                        - `erase_ratio`: Range of aspect ratio of erased area. *Default*: (1, 1).
                        - `erase_p`: Probability of applying random erase. *Default*=0.0
                        - `mixup_lambda`: min-max value of mixup strength. Default is 0-1. *Default*: `None`.
                        - `mixup_p`: Probability of applying random mixup v2. *Default*=0.0
                        - `input_key`: Can be `image` or `instance`. The input_key `instance` expects the KorniaAugmenter to follow the InstanceCropper else `image` otherwise for default.

- `model_config`: 
    - `init_weight`: model weights initialization method. "default" uses kaiming uniform initialization and "xavier" uses Xavier initialization method.
    - `backbone_config`:
        - `in_channels`:      Number of input channels. Default is 1.
        - `kernel_size`:      Size of the convolutional kernels. Default is 3.
        - `filters`:          Base number of filters in the network. Default is 32
        - `filters_rate`:     Factor to adjust the number of filters per block. Default is 1.5.
        - `down_block`s:      Number of downsampling blocks. Default is 4.
        - `up_blocks`:        Number of upsampling blocks in the decoder. Default is 3.
        - `convs_per_block`:  Number of convolutional layers per block. Default is 2.
    - `head_configs`
        - `head_type`:        Name of the head. Supported values are 'SingleInstanceConfmapsHead', 'CentroidConfmapsHead', 'CenteredInstanceConfmapsHead', 'MultiInstanceConfmapsHead', 'PartAffinityFieldsHead', 'ClassMapsHead', 'ClassVectorsHead', 'OffsetRefinementHead'
        - `head_config`:
            - `part_names`:   Text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used.
            - `anchor_part`:  index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
            - `sigma`: Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
            - `output_stride`: The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
            - `loss_weight`:  Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models.
- `trainer_config`: 
    - `train_data_loader`:
        - `batch_size`: number of samples per batch or batch size for training data. *Default* = 1.
        - `shuffle`: set to True to have the data reshuffled at every epoch. *Default*: False.
        - `num_workers`: number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. *Default*: 0.
        - `pin_memory`: Boolean  If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
        - `drop_last`: set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. *Default*: False.
        - `prefetch_factor`: Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. (default value depends on the set value for num_workers. If value of num_workers=0 default is None. Otherwise, if value of num_workers > 0 default is 2).
    - `model_ckpt`:
        - `save_top_k`: if save_top_k == k, the best k models according to the quantity monitored will be saved. if save_top_k == 0, no models are saved. if save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False.
        - `save_last`: When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. *Default*: None.
        - `auto_insert_metric_name`– When True, the checkpoints filenames will contain the metric name. For example, `filename='checkpoint_{epoch:02d}-{acc:02.0f}` with epoch 1 and acc 1.12 will resolve to `checkpoint_epoch=01-acc=01.ckpt`.
        - `every_n_epochs`: Number of epochs between checkpoints. This value must be None or non-negative. To disable saving top-k checkpoints, set every_n_epochs = 0. This argument does not impact the saving of save_last=True checkpoints. If all of every_n_epochs, every_n_train_steps and train_time_interval are None, we save a checkpoint at the end of every epoch (equivalent to every_n_epochs = 1). If every_n_epochs == None and either every_n_train_steps != None or train_time_interval != None, saving at the end of each epoch is disabled (equivalent to every_n_epochs = 0). This must be mutually exclusive with every_n_train_steps and train_time_interval. Setting both ModelCheckpoint(..., every_n_epochs=V, save_on_train_epoch_end=False) and Trainer(max_epochs=N, check_val_every_n_epoch=M) will only save checkpoints at epochs 0 < E <= N where both values for every_n_epochs and check_val_every_n_epoch evenly divide E.
        - `monitor`: Quantity to monitor. When None, this saves a checkpoint only for the last epoch. *Default*: None
        - `mode`: One of {min, max}. If save_top_k != 0, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For 'val_acc', this should be 'max', for 'val_loss' this should be 'min', etc. 
        - `filename`: Checkpoint filename which contains named formatting options to be auto-filled. When `None`, the filename will be set to '{epoch}-{step}', where “epoch” and “step” match the number of finished epoch and optimizer steps respectively. *Default*: None
    - `device`: device on which torch.Tensor will be allocated. One of the (cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu).
    - `trainer_devices`: Number of devices to train on (int), which devices to train on (list or str), or "auto" to select automatically.
    - `trainer_accelerator`: One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises the machine the model is running on and chooses the appropriate accelerator for the `Trainer` to be connected to.
    - `enable_progress_bar`: Boolean to enable printing the logs during training.
    - `max_epochs`: maxinum number of epochs to run.
    - `seed`: seed value for the current experiment.
    - `use_wandb`: Boolean to enable wandb logging.
    - `wandb_mode`: "offline" if only local logging is required.
    - `save_ckpt`: Boolean to enable checkpointing. 
    - `save_ckpt_path`: dir path to save the .ckpt file. *Default*: "./"
    - `wandb`:
        - `project`: title for the wandb project
        - `name`: name of the current run
        - `api_key`: API key
    - `optimizer`
        - `lr`: learning rate of type float. *Default*: 1e-3
        - `amsgrad`: Boolean to enable AMSGrad. *Default*: False
    - `lr_scheduler`
        - `mode`: One of "min", "max". In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. *Default*: "min".
        - `threshold`: Threshold for measuring the new optimum, to only focus on significant changes. *Default*: 1e-4.
        - `threshold_mode`: One of "rel", "abs". In rel mode, dynamic_threshold = best * ( 1 + threshold ) in max mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. *Default*: "rel".
        - `cooldown`: Number of epochs to wait before resuming normal operation after lr has been reduced. *Default*: 0
        - `patience`: Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the third epoch if the loss still hasn’t improved then. *Default*: 10.
        - `factor`: Factor by which the learning rate will be reduced. new_lr = lr * factor. *Default*: 0.1.
        - `min_lr`: A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. *Default*: 0.