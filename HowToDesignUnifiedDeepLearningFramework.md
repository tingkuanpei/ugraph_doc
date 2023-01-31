# 如何设计统一的深度学习框架

深度学习框架是基于 Tensor 和 Operator 构建的计算框架。Tensor 表示 0-N 维数组，包括 0 维的标量（Scalar）、1 维的向量（Vector）和多维的张量（Tensor）。Operator 类似于函数，描述在 Tensor 上执行的计算，Operator 输入 0-M 个 Tensor，输出 0-N 个 Tensor。基于 Tensor 和 Operator 构建的计算框架，可以用于很多问题的求解，如矩阵计算、图像处理、图形渲染等。而深度学习框架专注于深度学习领域。和其他领域相比，深度学习领域具有以下特点：

1. 使用梯度下降算法求最优解。这要求计算框架支持自动微分（特别是求一阶导数）和梯度下降算法。

2. 深度学习网络的计算量非常大，且需要运行在不同场景（云端、桌面端、边缘设备、移动端）和不同种类的计算设备上（CPU、GPU、DSP、NPU等）。不同的场景和设备，功耗不同、提供的算力不同，但都要求框架充分发挥计算设备的计算性能，提供尽可能快的计算速度。

3. 深度学习中的不同子领域使用的深度学习算法不同，且还在不断迭代演进。在 CV 领域，通常使用基于卷积的网络；在 NLP 领域则使用 RNN、Transformer等；而在推荐系统领域，则更流行 Embedding + DNN 分类器的算法。深度学习框架在支持各个领域的经典算法的同时，还需跟进业内最新的研究进展。

深度学习领域的特点，给计算框架的设计带来了一系列的挑战：

1. 训练接口的易用性。深度学习领域的技术突破来源于算法工程师。算法工程师要求训练框架足够易用，包括文档齐全、模型库完备，易于修改和调试，以便能够不断尝试新的网络结构。

2. 部署的易用性。深度学习算法往往只是整个产品的其中一环，算法需要嵌入各式各样的应用中，部署环境非常多样。常用的部署环境有：1.后台(Linux) CPU(X64)、GPU(NVIDIA) 单机服务器和分布式计算集群；2. 桌面(Windows/Linux/macOS) CPU(X64/X86/Arm64) 和 GPU(NVIDIA/AMD/AppleSilicon) 设备；3. 边缘(嵌入式Linux/Android) CPU(Arm64)、GPU、DSP、NPU 和 ASIC 设备。4. 移动端(Android/iOS/嵌入式Linux) CPU(Arm32/Arm64)、GPU(Mail/Adreno/PowerVR/AppleSilicon)、DSP 和 NPU 设备。在不同环境的不同计算设备上，框架需要保证计算结果一致。框架还需要提供多种编程语言的API：Python、C++、JAVA、Objective-C 等，以便不同领域的工程师可以直接调用。

3. 高性能。由于深度学习算法大多计算或访存密集，因此会消耗大量的计算资源。框架在计算设备上优化得越好，意味着算法工程师可以使用计算量更大、效果更优的模型。为了取得最佳的计算性能，工程师通常会直接使用汇编语言针对每一个 Operator 精心设计代码，这需要投入大量的资深工程师，并付出大量的努力。当单台计算设备不足以满足需求时，还会使用分布式计算集群：NLP预训练大模型训练时，会使用几百台 GPU 超级计算机组合成的系统训练网络；在推荐系统中，会使用几十台 CPU 计算机组成的系统，存储、更新、索引 Embedding 表。深度学习框架不仅需要在单机设备上具有优异的性能，还需要在分布式计算集群上表现优异。

为了解决这些挑战，涌现出了很多框架。[Caffe](https://caffe.berkeleyvision.org/) 借助于 NVIDIA GPU 的高算力，提供了第一个可用的深度学习框架。[TensorFlow](https://github.com/tensorflow/tensorflow) 以 Python 为前端，算法工程师不再需要接触复杂的 c++ runtime 即可训练模型，这极大地降低了开发门槛；同时它还提供了一套较高性能的从训练到部署的解决方案，这使得 TensorFlow 彻底淘汰了 Caffe。[PyTorch](https://github.com/pytorch/pytorch) 则从易用性出发，基于动态图构建框架，进一步降低了算法工程师修改和调试网络的门槛。凭借其易用性，PyTorch 越来越受到算法工程师的欢迎。Caffe -> TensorFlow -> PyTorch 基本解决了深度学习模型训练的问题，但是到了模型部署阶段性能往往无法达到最优。[ncnn](https://github.com/Tencent/ncnn)、[TensorRT](https://github.com/NVIDIA/TensorRT)、[oneDNN](https://github.com/oneapi-src/oneDNN)、[MNN](https://github.com/alibaba/MNN) 等框架，用于解决在不同计算设备上的模型部署问题。随着模型算量越来越大，单台机器已经无法满足模型训练和部署的需求，[TensorFlow-PS](https://www.tensorflow.org/tutorials/distribute/parameter_server_training)、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[OneFlow](https://github.com/Oneflow-Inc/oneflow)、[HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR) 等探索了超大模型的训练部署难题。

不同的框架只能解决某一部分问题，算法从训练到部署的整个流程中，工程师需要往返于不同的框架，这极大地提高了使用门槛。同时不同的框架中存在大量相同功能的冗余代码，这也降低了开发的效率。UGraph 希望引入一些技术，尝试在统一的架构下解决这些挑战。UGraph 中的 U 是 unify 的缩写，UGraph 致力于实现几方面的统一：1. 动态图和静态图的统一；2. 训练和推理的统一；3. 各种芯片调用的统一；4. 单机计算和分布式计算的统一。

## 1. 动态图和静态图的统一

Tensor 和 Operator 的构建和执行序列称为计算序列。计算序列的描述方式有两种：1.静态图模式和 2.动态图模式。静态图模式是 TensorFlow 1.x 中的使用方式，它先描述 Tensor 和 Operator 构成的计算图，再基于计算图执行计算。动态图模式是 PyTorch 中使用的模式，它直接定义 Tensor 并在 Tensor 上调用 Operator。动态图模式简洁直观、易于调试和修改。静态图模式在执行前可以对整个计算图进行优化，性能更优。两种模式各有优劣，互相不可替代，深度学习框架需要同时支持这两种模式，实现动态图与静态图的统一。

动态图与静态图的统一包含两个层次：1. 动态图和静态图 runtime 的统一；2. 动态图与静态图接口的统一。

- 动态图和静态图 runtime 的统一

现有的框架(PyTorch、TensorFlow、PaddlePaddle等)在开发之初往往只实现了动态图和静态图中其中一种模式，而后期又不得不同时支持两种模式，这导致它们不得不使用两套独立的 runtime 分别支持动态图和静态图。这极大地提高了开发成本，且后续开发的新特性需要分别支持两种模式，如量化、混合精度和数据并行等。更糟糕的是，用户可能在一份代码中同时使用了动态图模式和静态图模式，在这两种模式过渡的边界，需要处理很多 corner case：比如 Tensor 何时在何地释放、自动微分推导如何跨越两种模式等。

- 动态图与静态图接口的统一

PyTorch 和 TensorFlow 都在尝试为动态图和静态图提供统一的接口，如 PyTorch 2.x 中提供的 `torch.compile()` 接口；TensorFlow 2.x 中提供的 `tf.function()` 接口。这些接口是为了方便用户将动态图执行的网络转换为静态图执行，以获得更好的执行性能。动态图转换为静态图，首先要解决如何从动态图中获取图的结构的问题，常用的方式有 JIT 和 graph trace 两种，它们各有优劣。在设计动态图和静态图转换的接口时，还需要兼顾动态图的灵活性和静态图的高性能。

### 1.1. “Graph” 架构

为了统一动态图和静态图的 runtime，UGraph 提出了 “Graph” 架构，其核心想法是：用户使用动态图模式描述计算序列时，不需要立即执行计算序列，当 Tensor 被取值时，返回对应的 Tensor 的值即可。

在实现中，UGraph 使用了异步计算的思想，分为构图线程和异步计算线程。用户在构图线程上创建计算图的计算节点，并将计算节点存入共享队列中。计算节点上会描述计算序列的详细信息，如 Tensor 的 Shape、DataKind 和如何执行计算等，但不包含 Tensor 的数据，也不执行计算。在大多数情况下，用户都不需要访问 Tensor 的值，因此构图线程不需要等待异步线程，可以源源不断地将计算节点存入共享队列中。异步线程会从共享队列中读取当前累计的计算节点，将计算节点组成计算图，并执行计算。当构图线程需要访问 Tensor 的值时，会触发构图线程与异步线程之间的同步，并等待异步线程将 Tensor 的值发送回来。构图线程还会将 Tensor 是否被析构的信息发给异步线程，异步线程会持有相应的 Tensor 直至 Tensor 被用户析构，因此重复访问同一 Tensor 的值，不会导致重复计算。UGraph 的动态图模式和静态图模式，都使用了基于计算图的 runtime，这极大地降低了代码的开发和维护成本。

UGraph 使用逻辑计算图记录构图线程上的计算节点。逻辑计算图具有完备性，可以描述执行计算序列所需的全部信息，并且可以与具体的执行过程一一对应，但其不包含真正执行的计算所需的计算资源。逻辑计算图包含的信息包括图的拓扑结构，Tensor 节点的 shape、layout 等参数，Operator 节点的参数、在哪个计算流上执行，计算流如何并发及同步等。

## 1.2. 异步计算与图改写

"Graph" 的架构使得 UGraph 拥有两项至关重要的特性：1.异步计算；2.图改写。

- 异步计算

UGraph 中，动态图的构图与计算异步执行，当用户获取 Tensor 的值时，才会触发隐式的同步操作。这使得动态图的 runtime 可以拿得“未来”的计算图，从而获得更多的优化机会：如动态图中生命周期不重叠的 Tensor 共享内存；在分布式系统中，根据计算图，降低数据的传输量，同时尽可能 overlap 数据传输和计算。

- 图改写

在计算的过程中，用户只关心输出 Tensor 的值，因此可以对计算图进行修改，以获得更好的计算性能。由于动态图模式可以拿到 “未来” 的计算图，静态图模式天然地存在完整的计算图，因此可以基于图改写，可以以统一的方式支持很多高级特性：

1. 删除多余的节点：用户在描述计算序列时，可能生成某些 Tensor，但又没有获取这些 Tensor 的值，因此可以在计算图中将这些 Tensor 相关的计算序列移除。

2. 算子融合：在推理部署时，一部分算子可以融合成一个数值等价的算子。如 Convolution 和 Batch Normalization 可以融合为 Convolution。

3. 自动微分：基于前向图，生成反向图的节点。

4. 混合精度与量化：自动转换 Tensor 的 DataKind，以支持 fp32-fp16 混合精度和 int8 量化计算。

5. DistributedTensor：DistributedTensor 指的是分布在多台计算机上的 Tensor。基于图改写，用户在不修改构图代码的情况下，也可以支持 DistributedTensor。

## 1.3. 动态图与静态图接口的统一

动态图与静态图之间转换和使用接口如何设计，还有待进一步的研究。目前 UGraph 中有两种方式可以实现动态图与静态图的转换：1. functional 函数接口通过传参来区分；2. graph trace。UGraph 中使用类似于 PyTorch 的 `torch.nn.functional` 的 functional 函数接口构图，每一个函数都可传入 graph 参数，用于指定当前使用的 graph，如果没有指定，则为 thread local 的 eager graph。graph trace 功能，则可将动态图的计算序列记录下来，再保存成静态图。

UGraph 中，动态图和静态图的 Tensor 可以混合使用，比如动态图的 Tensor 可以作为静态图的输入 Tensor，静态图的输出 Tensor 可以用于动态图的计算中。静态图可以作为动态图的一个子图执行，基于此，可以支持动态图和静态图混合计算时的自动微分。

## 2. 训练和推理的统一

深度学习模型训练时，需要运行前向计算和反向计算；而在模型推理部署阶段，只需运行前向计算。但是以往的深度学习训练框架，由于设计上的原因，无法完全满足模型部署时推理的需求，如 TensorFlow 推荐使用 TensorFlow Lite 进行移动端模型推理；PyTorch 模型通常需要转换为 ONNX 模型，再借助其他推理框架部署。模型训练与部署存在几点差异，这些差异导致了只为模型训练设计的框架，无法完全满足模型推理的需求：

1. 运行的场景和芯片不同。模型训练通常在后台 GPU 服务器上运行，而模型推理则需要部署在各种操作系统和各种芯片上，不同芯片的调用方式存在差异，推理框架需要为不同的芯片提供统一的调用接口。和 GPU 高性能服务器相比，在移动端设备上，受限于芯片的功耗，计算芯片提供的算力、内存、存储空间都非常有限，这对推理框架提出了更为苛刻的要求。

2. 对框架的开销敏感。模型训练时，通常使用大 Batch 进行训练，Operator 的计算开销远大于框架的调度开销。而在模型推理时，通常是单 Batch，网络的计算量更小。在移动端设备上，由于芯片的算力有限，有时只能使用计算量为 MB FLOPS 量级的网络，此时框架的调度开销无法忽略。

3. 需要嵌入不同的应用中。训练框架只需提供 Python 语言的前端，核心功能可以使用 Python 语言编写，并使用 Python 作为动态图的描述语言。推理框架由于部署环境多样，不一定有 Python 环境，因此需要支持从磁盘中加载静态图。由于跨平台的需求，推理框架核心部分必须使用 C/C++ 语言编写，并提供 JAVA、Objective-C 等语言的接口。

UGraph 同时支持动态图和静态图，在模型训练阶段，用户可以使用动态图和静态图进行模型训练，而在模型部署阶段，则可以使用静态图进行部署。UGraph 设计了一套轻量化、兼容多芯片、支持并发的计算图执行引擎，可以同时满足模型训练和模型部署的需求。

### 2.1. Kernel 运行时

参照 CUDA stream，UGraph 提出了由 Tensor、Kernel 和 ComputeStream 组成的 Kernel 运行时为不同的芯片提供统一的计算抽象。

Tensor 表示张量，会持有分配好的内存。并支持直接分配内存或从内存池中分配内存。Tensor 还提供一套基础的计算接口，如矩阵乘法、转置、加法等，方便开发者实现新的 Kernel。

Kernel 对输入 Tensor 执行计算，并将计算结果写到输出 Tensor 中。Kernel 提供了 Compile() 和 Compute() 两个接口，与输入 Tensor 的内容无关的操作，可以提前到 Compile 阶段完成。

ComputeStream 对计算芯片提供的 CPU 线程、CUDA Stream 和 OpenCL Command Queue 等提供了统一的抽象。Kernel 必须在 ComputeStream 上执行，不同 ComputeStream 之间的计算相互独立，但可以通过 Event 进行同步。ComputeStream 还分为Sync Stream（在当前 CPU 线程执行 Kernel）和 Async Stream（在独立的 CPU 线程上执行 Kernel），可以满足模型训练时的计算并发及模型部署推理时的低开销的需求。

### 2.2. 内存规划和内存池

通常情况下，Tensor 的内存分配时机为 Kernel 执行前，并使用引用计数管理 Tensor。但在 UGraph 中，基于计算图，可以提前分配计算图中所有 Tensor 的内存，并保证生命期不同的 Tensor 共享同一块内存，这种分配方式被称为内存规划。内存规划可以降低模型计算时对内存占用。内存规划还使得内存的申请和释放，可以从 Kernel 的运行期提前到 Kernel 的编译期，从而提高 Kernel 的执行速度。当 Tensor 在多个 ComputeStream 上被使用时，也可根据计算图，在正确的时机释放 Tensor。

为了降低内存碎片和从操作系统(GPU 驱动)中反复申请内存的开销，UGraph 使用内存池管理分配到的内存。

### 2.3 编译期和运行期

为降低运行时框架的开销，UGraph 将计算图的执行严格区分为编译期和运行期。在动态图模式中，编译期和运行期在不同线程上并发执行。在静态图模式中，编译期只在计算图创建时执行一次，之后只需反复调用运行期即可。计算图的创建、图改写、计算资源的创建和分配、内存规划和内存分配和 Kernel 的 Compile() 接口等都属于编译期，运行期只包括 Kernel 的 Compute()。这种做法可以最大限度降低框架的调度开销对计算延时的影响。

## 3. 各种芯片调用的统一

深度学习框架需要运行在各种芯片上，包括：X86_64/Arm CPU、NVIDIA/AMD/Mail/Adreno/PowerVR/AppleSilicon GPU、DSP、NPU 等。这些芯片的调用，存在以下差异：

1. 编程环境不同。在 CPU 上，可以直接使用 C/C++ 或使用 X86_64/Arm 汇编语言编程；在 GPU 环境，则需要使用 CUDA/OpenCL 与 GPU 驱动交互；DSP 和 NPU 通常直接调用芯片厂商提供的计算库。

2. 不同芯片的最佳的 Tensor Layout、DataKind 不同。以常用的卷积运算为例，在 Intel CPU 上可能使用 nChw16c 的 Layout：[oneDNN 文档](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html#blocked-layout)；在 NVIDIA 的 GPU 上可能使用 NHWC 的 Layout：[cuDNN 文档](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#nhwc-layout-x32)。

3. 可能需要一次执行整个网络。如 NVIDIA 提供的 TensorRT，需要对整个计算图进行编译并执行计算，不支持逐个 Operator 调用。

### 3.1 再谈 Kernel 运行时

UGraph 中的 Tensor 为 CPU、GPU 的不同 Layout、DataKind 的 buffer memory 或 texture memory 等提供了统一的抽象。

Kernel 中支持使用 CPU 汇编、CUDA Kernel、OpenCL Kernel 编程或者直接调用硬件厂商或第三方提供的 primitive 库，如 cuDNN、cuBLAS、oneDNN、Egien 等。Kernel 也支持为不同的芯片、不同的 Layout、DataKind 的 Tensor 定制不同的 Kernel。

配合图改写，UGraph 可以将整个网络或网络的一部分划分为一个子图 Operator，并在 Kernel 中计算一个子图。基于这种方式，可以支持类似 TensorRT、SNPE 等只能计算整个计算图的库。

### 3.2 TensorProxy 机制

不同芯片最佳的 Tensor Layout 不同，为了获得最佳的性能，框架必须在计算时，自动转换 Tensor 的 Layout，在最佳的 Layout 下执行计算。由于静态图模式可以获取整个计算图，因此可以轻松支持这种特性。而在动态图下，这并不容易。PyTorch 的动态图由于不支持图改写，因此它不得不让用户使用特定的 Tensor Layout 构图，如 [mkldnn tensor](https://pytorch.org/docs/stable/generated/torch.Tensor.to_mkldnn.html?highlight=torch+mkldnn)。这种方式需要改变构图的代码，对用户的要求过高。

UGraph 中则使用 TensorProxy 机制解决这一问题：基于图改写，自动转换 Tensor 的 Layout，在最佳的 Tensor Layout 下执行计算。当用户读取某个 Tensor 的值时，才会将 Tensor 转换为初始的 Layout 供用户访问。TensorProxy 不需要用户手动修改模型结构的描述代码，而是借助于图改写的能力，在动态图模式和静态图模式下以统一的方式支持。同时由于动态图模式异步计算的特性，在支持 TensorProxy 时，UGraph 仅在必要的节点前后转换 Tensor 的 Layout，不会产生多余的转换操作。

### 3.3 深度学习编译器

深度学习框架需要运行在不同的芯片上，针对每一款芯片，每一个 Operator 都使用汇编语言精心设计代码，需要投入大量的人力。深度学习编译器借鉴了高级语言编译器的设计经验，希望通过分析的方式，自动生成对应的汇编代码。目前比较流行的深度学习编译器有：TVM、TensorFlow XLA等。但它们目前都还无法完全替代人工精心设计的汇编代码。

UGraph 在架构设计时，考虑到了深度学习编译器的需求。UGraph 定义了一套完善的 IR，它可以通过编译器提供的工具转换为编译器的 IR。编译器的编译过程可以离线完成，也可在 Kernel::Compile() 接口中在线完成。编译器编译好的汇编代码可以在 Kernel::Compute() 接口中调用。

## 4. 单机计算和分布式计算的统一

类似 [TensorFlow-DTensor](https://www.Tensorflow.org/guide/dtensor_overview) 和 [PyTorch DistributedTensor](https://github.com/pytorch/pytorch/issues/88838)，UGraph 也将支持 DistributedTensor。由于 “Graph” 架构动态图和静态图统一的特性，UGraph 可以基于图改写，以统一的方式支持 DistributedTensor，并实现数据并行、模型并行、流水并行、自动并行、重计算等高级特性。

在动态图模式下，由于“Graph”架构可以拿到“未来”的计算图，因此可以为分布式计算提供更多的优化机会，如：分析 DistributedTensor 如何分布在不同的 GPU 上，以减小在不同 GPU 之间传输的数据的规模；不同 GPU 之间数据的传输与计算重叠等。

### 4.1. 并发模型

UGraph 支持多 ComputeStream，Kernel 在不同的 ComputeStream 上执行，即可实现 Kernel 的并发。为充分利用计算设备的并发潜能，UGraph 在框架层面支持不同层次的并行：1. Graph 间；2. Operator 间；3. Operator 内。

在 UGraph 中，不同 Graph 在不同线程上执行。因此，当用户需要并发时，创建多个 Graph 实例即可，不需要手动创建、管理线程。Graph 间暂时不提供同步操作，但提供 Tensor 的线程安全阻塞队列，典型的使用方式如下：在数据预处理时，创建多个 Graph 进行数据的预处理，将处理好的 Tensor 放入阻塞队列中，Main Graph 从阻塞队列中读取 Tensor，并执行模型训练。

在同一个 Graph 内，将不同 Operator 分配到不同的 ComputeStream 上，即可实现 Operator 间的并行。在计算图执行前，UGraph 会根据配置，为每个 Operator 分配最优的 ComputeStream。基于此功能，可以实现多线程并发计算、数据传输与计算重叠、多 GPU 并发等功能。

同一个 Operator 内，支持几种类型的并行：1. 将一个 Operator 的计算并发到不同的 ComputeStream 上完成，即 DistributedTensor；2. 在 CPU 上使用线程池计算 for loop；3. 计算设备提供的 SIMD、SIMT、MIMD 和数据传输与计算重叠能力。

基于 UGraph 提供的并发模型，可以实现不同的并发方式。以流水并行为例，可以将流水并行的不同子图划分到不同的 ComputeStream (不同CPU 线程或不同 GPU 计算卡) 上并发执行，并使用 Tensor 的固定容量的阻塞队列在不同 ComputeStream 之间传递 Tensor。Tensor 的阻塞队列分为 PushQueueOp 和 PopQueueOp 两种 Operator。前面的计算流将结果 push 进 PushQueueOp 中，如果队列满则阻塞；后面的计算流从 PopQueueOp 里 pop 结果并执行计算，如果队列空则阻塞。DistributedTensor 中经常需要进行 AllReduce 等操作，将 AllReduce Operator 和计算 Operator 划分到不同的 ComputeStream，即可实现数据传输与计算的重叠。

### 4.2. single-client to multi-worker 模式

UGraph 在分布式系统中使用 single-client to multi-worker 的模式：用户通过 single-client 构建计算节点，进行图改写，再将计算分发到 multi-worker 上执行。由于 UGraph 构图与计算分离和异步计算的特性，大部分时间 multi-worker 不需要与 single-client 进行同步，只需接收 single-client 传输过来的计算图并执行对应的计算即可，这保证了分布式系统上的高性能。

在动态图模式，当用户需要获取 Tensor 的值时，会阻塞 single-client，从而阻塞 multi-worker。UGraph 还提供了一些机制，避免此类阻塞的发生：1. 异步获取 Tensor 值；2. 异步设置 Tensor 值。

- 异步获取 Tensor 值
深度学习的模型训练过程中，需要定期打印 loss，这需要访问 loss Tensor 的值。因此 UGraph 将提供类似于 C++ future 或异步回调的机制，实现异步打印 loss。

- 异步设置 Tensor 值
在反向传播时，需要设置 loss.grad 为 1 或特定值。UGraph 将提供机制，异步设置 loss.grad。

虽然 UGraph 提供了一些机制支持异步获取和设置 Tensor 的值，但依旧无法避免 single-client 和 multi-worker 的阻塞。比如 Tensor 的值决定了后续模型的结构时，就必须等待 Tensor 计算完成后，才能构建后续的计算序列。UGraph 会以 Warning 的形式，提醒用户避开这类型的操作。
