import hydra
from omegaconf import DictConfig, OmegaConf
from sleap_nn.training.model_trainer import train


@hydra.main(version_base=None, config_path="", config_name="initial_config")
def main(cfg: DictConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    train(
        train_labels_path=cfg.data_config.train_labels_path,
        val_labels_path=cfg.data_config.val_labels_path,
        user_instances_only=cfg.data_config.user_instances_only,
        data_pipeline_fw=cfg.data_config.data_pipeline_fw,
        np_chunks_path=cfg.data_config.np_chunks_path,
        litdata_chunks_path=cfg.data_config.litdata_chunks_path,
        use_existing_chunks=cfg.data_config.use_existing_chunks,
        chunk_size=cfg.data_config.chunk_size,
        delete_chunks_after_training=cfg.data_config.delete_chunks_after_training,
        is_rgb=cfg.data_config.preprocessing.is_rgb,
        scale=cfg.data_config.preprocessing.scale,
        max_height=cfg.data_config.preprocessing.max_height,
        max_width=cfg.data_config.preprocessing.max_width,
        crop_hw=cfg.data_config.preprocessing.crop_hw,
        min_crop_size=cfg.data_config.preprocessing.min_crop_size,
        use_augmentations_train=cfg.data_config.use_augmentations_train,
        intensity_aug=dict(cfg.data_config.augmentation_config.intensity),
        geometry_aug=dict(cfg.data_config.augmentation_config.geometric),
        init_weight=cfg.model_config.init_weights,
        pre_trained_weights=cfg.model_config.pre_trained_weights,
        pretrained_backbone_weights=cfg.model_config.pretrained_backbone_weights,
        pretrained_head_weights=cfg.model_config.pretrained_head_weights,
        backbone_config=dict(cfg.model_config.backbone_config),
        head_configs=dict(cfg.model_config.head_configs),
        batch_size=cfg.trainer_config.train_data_loader.batch_size,
        shuffle_train=cfg.trainer_config.train_data_loader.shuffle,
        num_workers=cfg.trainer_config.train_data_loader.num_workers,
        ckpt_save_top_k=cfg.trainer_config.model_ckpt.save_top_k,
        ckpt_save_last=cfg.trainer_config.model_ckpt.save_last,
        trainer_num_devices=cfg.trainer_config.trainer_devices,
        trainer_accelerator=cfg.trainer_config.trainer_accelerator,
        enable_progress_bar=cfg.trainer_config.enable_progress_bar,
        steps_per_epoch=cfg.trainer_config.steps_per_epoch,
        max_epochs=cfg.trainer_config.max_epochs,
        seed=cfg.trainer_config.seed,
        use_wandb=cfg.trainer_config.use_wandb,
        save_ckpt=cfg.trainer_config.save_ckpt,
        save_ckpt_path=cfg.trainer_config.save_ckpt_path,
        resume_ckpt_path=cfg.trainer_config.resume_ckpt_path,
        wandb_entity=cfg.trainer_config.wandb.entity,
        wandb_project=cfg.trainer_config.wandb.project,
        wandb_name=cfg.trainer_config.wandb.name,
        wandb_api_key=cfg.trainer_config.wandb.api_key,
        wandb_mode=cfg.trainer_config.wandb.wandb_mode,
        wandb_resume_prv_runid=cfg.trainer_config.wandb.prv_runid,
        wandb_group_name=cfg.trainer_config.wandb.group,
        optimizer=cfg.trainer_config.optimizer_name,
        learning_rate=cfg.trainer_config.optimizer.lr,
        amsgrad=cfg.trainer_config.optimizer.amsgrad,
        lr_scheduler=dict(cfg.trainer_config.lr_scheduler),
        early_stopping=cfg.trainer_config.early_stopping.stop_training_on_plateau,
        early_stopping_min_delta=cfg.trainer_config.early_stopping.min_delta,
        early_stopping_patience=cfg.trainer_config.early_stopping.patience,
    )

    # run inference on val (and test) set (predict on test data if test data path is provided)


if __name__ == "__main__":
    main()
