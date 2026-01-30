# SLEAP-NN

<div class="hero" markdown>
![SLEAP pose estimation demo](assets/sleap_movie.gif)
</div>

<div class="badges" markdown>
[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![GitHub stars](https://img.shields.io/github/stars/talmolab/sleap-nn)](https://github.com/talmolab/sleap-nn)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-nn)
[![PyPI](https://img.shields.io/pypi/v/sleap-nn?label=PyPI)](https://pypi.org/project/sleap-nn)
</div>

**SLEAP-NN** is the PyTorch backend for [SLEAP](https://sleap.ai), providing neural network training and inference for multi-animal pose estimation. It offers an end-to-end workflow from labeled data to tracked predictions, with seamless integration into SLEAP's GUI and command-line tools.

## ✨ Features

- **Multiple model types** – Single instance, top-down, bottom-up, and centroid models
- **Modern backbones** – UNet, ConvNeXt, and Swin Transformer architectures
- **Multi-GPU training** – PyTorch Lightning with DDP support
- **Production export** – ONNX and TensorRT for fast deployment
- **Flexible config** – Hydra/OmegaConf for reproducible experiments

**Let's start SLEAP-NNing!** 🐭🐭

---

## 📚 Explore the Docs

<div class="grid cards" markdown>

-   🚀 **Quick Start**

    ---

    Install and train your first model in 5 minutes.

    [:octicons-arrow-right-24: Get Started](getting-started/quickstart.md)

-   📖 **Tutorials**

    ---

    Quick start, first model tutorial, and example notebooks.

    [:octicons-arrow-right-24: Learn](tutorials/index.md)

-   📚 **Guides**

    ---

    Training, inference, and tracking workflows.

    [:octicons-arrow-right-24: Explore](guides/index.md)

-   ⚙️ **Configuration**

    ---

    Customize your training config.

    [:octicons-arrow-right-24: Configure](configuration/index.md)

-   💻 **CLI Reference**

    ---

    Command-line interface documentation.

    [:octicons-arrow-right-24: Commands](reference/cli.md)

-   🔧 **API Reference**

    ---

    Full Python API documentation.

    [:octicons-arrow-right-24: API Docs](api/)

</div>

---

## 🔄 Coming from SLEAP < v1.5?

[:octicons-arrow-right-24: Load legacy models for inference](guides/inference.md#legacy-sleap-model-support)

[:octicons-arrow-right-24: Convert legacy config](configuration/index.md#converting-from-legacy-sleap)

| SLEAP < v1.5 | SLEAP-NN |
|-----------|----------|
| TensorFlow/Keras | PyTorch/Lightning |
| Single GPU | Multi-GPU (DDP) |
| Limited export | ONNX + TensorRT |



---

## Get Help

<div class="grid cards" markdown>

-   :material-frequently-asked-questions:{ .lg } **FAQ**

    Common questions answered. [View FAQ](help/faq.md)

-   :fontawesome-brands-github:{ .lg } **Report Issues**

    Found a bug? [Create an issue](https://github.com/talmolab/sleap-nn/issues/new)

-   :material-forum:{ .lg } **Discussions**

    Questions? [Start a discussion](https://github.com/talmolab/sleap-nn/discussions)

</div>
