# Configuration

SLEAP-NN uses YAML configuration files with three main sections.

| Section | Description |
|---------|-------------|
| [Data Config](data.md) | Data paths, preprocessing, augmentation |
| [Model Config](model.md) | Backbone and head architecture |
| [Trainer Config](trainer.md) | Epochs, optimizer, learning rate, logging |
| [Sample Configs](samples.md) | Ready-to-use templates |

---

## Config Structure

```yaml
data_config:      # Data loading and preprocessing
  train_labels_path: [...]
  preprocessing: {...}
  augmentation_config: {...}

model_config:     # Model architecture
  backbone_config: {...}
  head_configs: {...}

trainer_config:   # Training settings
  max_epochs: 100
  optimizer: {...}
```

---

## Usage

```bash
sleap-nn train --config config.yaml
```

---

## Converting from Legacy SLEAP

If you have a `training_config.json` from SLEAP < v1.5, you can convert it to the new YAML format:

```python
from sleap_nn.config.training_job_config import TrainingJobConfig
from omegaconf import OmegaConf

# Load legacy config and convert
config = TrainingJobConfig.load_sleap_config("path/to/training_config.json")

# Save as YAML
OmegaConf.save(config, "config.yaml")
```
