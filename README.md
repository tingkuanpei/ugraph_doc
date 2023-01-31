# UGraph

UGraph 是深度学习框架，UGraph 中的 U 是 unify 的缩写，UGraph 致力于实现几方面的统一： 1. 动态图和静态图的统一；2. 训练和推理的统一；3. 各种芯片调用的统一；4. 单机计算和分布式计算的统一。在此之前，这些功能都有不同的实现方式，并且需要使用不同的框架，比如 [TensorFlow](https://github.com/tensorflow/tensorflow)、[PyTorch](https://github.com/pytorch/pytorch)、[oneDNN](https://github.com/oneapi-src/oneDNN)、[TensorRT](https://github.com/NVIDIA/TensorRT)、[MNN](https://github.com/alibaba/MNN)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR) 等。UGraph 将探索深度学习框架中使用到的各类技术，尝试以统一的方式解决这些问题。

设计目标：

- **易用性**：模型训练：易于调试、修改；模型部署：易于嵌入各类应用中；框架开发：架构清晰、文档齐全、开发成本低。
- **高性能**：充分发挥计算设备提供的计算和并发潜能。
- **统一**：在统一的架构下，解决深度学习的各类问题，即降低框架的开发成本，也避免用户在不同框架之间迁移的成本。

## 架构设计

- [如何设计统一的深度学习框架](./HowToDesignUnifiedDeepLearningFramework.md)：介绍 UGraph 构建统一的深度学习框架使用的方法。
