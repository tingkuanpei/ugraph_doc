# Graph 概述

Tensor 和 Operator 的构建和执行序列称为计算序列。计算序列的描述方式有两种：1.静态图模式和 2.动态图模式。

静态图模式是 TensorFlow 1.x 中使用的模式。用户需要先定义计算图，再根据计算图执行计算。在计算执行前，可以对整张计算图进行分析，因此静态图模式拥有更好的性能。但是先构图再计算的方式不够灵活，用户在调试代码时，需要手动插入`watch`节点，才能访问 Tensor 的数据。为了解决静态图易用性的问题，TensorFlow 2.0 推出动态图的模式。TensorFlow 2.0 的动态图模式和静态图模式，运行的是两个不同的代码分支，因此在添加新功能时，两个模式都需要添加重复功能的代码，这极大地增加了代码的维护成本。

PyTorch 则默认使用动态图模式，通过 python 脚本直接调用 c++ 实现的计算函数，不需要预先定义计算图。由于 PyTorch 所见即所得的易用性，其逐渐获得算法工程师的青睐。动态图模式易用性好，但是性能较低，因此 PyTorch 2.x 新增了 `torch.compile()`，可以将动态图转为静态图。

动态图具有易用性的特点，静态图具有高性能、易部署的特点，分别支持两种模式会增加代码的开发与维护成本，因此 UGraph 提出并实现了“Graph”架构，希望以统一的方式，同时支持动态图与静态图。

## 1. “Graph”架构

“Graph”架构，基于计算图，以统一的方式实现了静态图和动态图的运行时。“Graph” 的核心思想是：动态图模式下，用户的 python 构图代码是对计算过程的描述，而不是计算本身。用户在描述计算序列时，不需要立刻执行计算。当 Tensor 被取值时，再返回 Tensor 的值给用户即可。

在实现中，使用了异步计算的思想，分为构图线程和异步计算线程。用户在构图线程上创建计算图的计算节点，并将计算节点存入共享队列中。计算节点上会描述计算序列的详细信息，如 Tensor 的 Shape、DataKind 和如何执行计算等，但不包含 Tensor 的数据，也不执行计算。在大多数情况下，用户都不需要访问 Tensor 的值，因此构图线程不需要等待异步线程，可以源源不断地将计算节点存入共享队列中。异步线程会从共享队列中读取当前累计的计算节点，将计算节点组成计算图，并执行计算。当构图线程需要访问 Tensor 的值时，会触发构图线程与异步线程之间的同步，并等待异步线程将 Tensor 的值发送回来。构图线程还会将 Tensor 是否被析构的信息发给异步线程，异步线程会持有相应的 Tensor 直至 Tensor 被用户析构，因此当重复访问同一 Tensor 的值时，不会重复计算。

在下面的 python 代码中，执行了`convolution`和`relu`两个计算。从 TensorFlow 静态图的角度看，需要先定义包含`convolution`和`relu`两个节点的计算图，再进行计算。而从 PyTorch 动态图的角度看，先执行第一个算子`convolution`获得 tensor y，再执行第二个算子`relu`获得 tensor t。而从 “Graph” 的角度看，python 线程，会将`convolution`和`relu`两个节点发送给异步线程。当执行到`print(t)`语句时，python 线程会告知异步线程，当前需要 tensor t 的值。异步线程会构建包含`convolution`和`relu`两个节点的计算流，并求取 tensor t 的值返回给 python 线程。

```python
def layer(x, weight):
	y = convolution(x, weight)
	t = relu(y)
  return t

x = Tensor()
weight = Tensor()
t = layer(x, weight)
print(t)
```

和 TensorFlow 与 PyTorch 相比，“Graph”具有以下优势：
1. 动态图和静态图都基于计算图实现，因此可以最大程度复用代码，降低了代码的开发和维护成本。
2. 由于在动态图模式下，也可以拿到“未来”的计算图。因此动态图和静态图都可以基于计算图，进行图改写，获得更好的计算性能，并支持混合精度、量化等高级特性。
3. “Graph”架构使用了异步计算的思想，可以为分布式系统提供更多的 overlap 数据传输与计算的机会。
4. 用户可以在动态图下修改调试模型，并转成静态图进行模型的训练及部署，这在最大程度地满足了用户易用性与高性能的需求。

## 2. 实现细节

### 2.1 . 静态图

静态图由⼏个组件实现：1. LogicalGraph：逻辑计算图，可以完备表示计算图中的节点及执行信息。2. GraphConstructor：⽤户可以使⽤ GraphConstructor，往 LogicalGraph 中逐个添加计算节点。3. GraphTransformer：对 LogicalGraph 进⾏图改写，实现混合精度、算⼦融合和转变 Tensor 的 Layout 等功能。4. ComputeResource：根据 LogicalGraph 分配内存，⽣成执⾏计算的 Kernel等。5. Executor：根据执行序列，执行计算并查询结果。

以上组件由 StaticExecutor 进行调度，其执行顺序为：1. 构建 LogicalGraph：使用 GraphConstructor 构建或从磁盘读取实现序列化的模型文件。2. 调用 GraphTransformer 执行图改写，以获得最佳的性能。3. 根据 LogicalGraph 分配所有的计算资源，即 ComputeResource。4. 生成 Operator 的执行序列，将执行序列发送给 Executor 执行并查询计算结果。

### 2.2. 动态图

动态图的实现在静态图的基础上多了两个组件：1. EagerGraphConstructor：派⽣⾃ GraphConstructor，当添加节点时，会将计算节点发送到异步线程。2. EagerExecutor：持有异步线程，接收到构图线程发送过来的计算节点后，会将当前接收到的所有计算节点，组合成 LogicalGraph，并调⽤ GraphTransformer 进⾏图改写，生成 ComputeResource 的节点，最后将当前的计算节点发送给 Executor 执⾏。

由于 EagerGraphConstructor 派生自 GraphConstructor。EagerExecutor 只是创建线程，并负责调度，因此可以最大程度复用静态图的代码，减小代码的开发和维护成本。

构图线程与异步线程之间使用消息队列进行构图线程到异步线程的单向的通讯，并使用 c++ std::future 获取返回值。

在 EagerExecutor 中，通过引⽤计数的⽅式，确定 Tensor 是否被构图线程持有，为内存复⽤、何时删除节点提供信息。当构图线程需要获取 Tensor 的值时，会通过 EagerGraphConstructor 向 EagerExecutor 发送信息，并等待直到 Tensor 计算完成。

EagerExecutor 中的异步线程会一直执行 while 循环，⽬前设计是当异步线程闲置时，立刻将当前接收到的节点组合成 LogicalGraph，并执⾏后续流程。由于构图线程执⾏得⽐异步线程更快，因此异步线程每⼀次执⾏的都是多个节点组合成的计算图。异步线程的执⾏时机需要更多的研究，原则是及时 launch kernel，避免 GPU 闲置，⼜能⼀次性执⾏尽可能多的节点，获得更好的性能。

EagerExecutor 中只执行必不可少的 GraphTransformer，以免延误 kernel 的 launch 时间。比如用户开启了混合精度的功能，则调用混合精度的 GraphTransform，若没有开启，则不会调用。

在动态图模式下，用户在 python 端写下的代码，和 PyTorch 是一样的，用户不会感知到计算图的存在。底层的计算引擎，会按照用户的要求，准确执行计算，并返回正确的 Tensor 的值。用户唯一需要知道的事情是，“Graph” 架构始终是异步执行，而 PyTorch 的 CPU 是同步执行，只有 GPU 是异步执行。

## 3. 为什么要统一动静态图的运行时

统一动静态图运行时最主要的原因是降低代码的开发成本。动态图的易用性和静态图易部署和高性能的特性都是用户需要的，因此一个完备的深度学习框架必须同时支持两种模式。两种模式使用独立的运行时，会极大地增加开发成本，如动态图基于 tape 的方式实现自动微分、而静态图则是基于前向计算图可直接生成反向图；动态图基于 JIT(如 PyTorch 的 TorchScript) 的方式支持混合精度和量化，而静态图基于图改写即可实现；动态图基于 multi-client 启动多进程支持数据并行（PyTorch 运行 DDP 需要用特殊的脚本启动多进程，在多个进程运行同一份 Python 脚本），而静态图只需要在内部创建多个计算流。当深度学习框架不断前进，需要支持越来越多功能时，同样的故事会反复出现，从而迫使开发者不断思考，为什么需要开发两份相同功能的代码。

“Graph”在计算图的基础上，构建了完整的动态图与静态图的运行时，这极大地降低了代码的开发与维护成本。“Graph”所支持的动态图与 PyTorch 的动态图相比，除了异步执行的特性外，用户不会感知到任何的差异。“Graph”所支持的静态图和 TensorFlow 的静态图设计理念功能基本一致。因此，“Graph”很好地统一了动态图和静态图的运行时。

## 4. 动态图与静态图的转换

“Graph”实现了动态图与静态图运行时的统一，但并未涉及动态图与静态图之间的转换。UGraph 提供了几种方式完成动态图与静态图之间的转换：
1. functional 和 Module 的构图接口，均支持指定当前使用的是动态图还是静态图。构图代码不变，传入不同种类的 graph 即可。
2. 动态图支持 Tracing 功能，把动态图的执行序列保存为静态图。
3. 静态图可以加载动态图保存的 Module.state_dict() 模型文件。
4. 未来还可通过 JIT 的方式，分析动态图 Python 构图代码的 AST，根据 AST 直接生成静态图。

## 5. 相关工作

“Graph”架构主要受到 [LazyTensor](https://arxiv.org/abs/2102.13267) 的启发。但 LazeTensor 仅局限在动态图模式和深度学习编译器的结合上，“Graph”则是基于计算图构建了完整的动态图和静态图的运行时。

其他类似的工作还有：TensorFlow eager runtime、PyTorch TorchScript 和 OneFlow eager runtime 等，它们都是使用 JIT 或者 Tracing 的方式实现动态图转静态图，并未统一动态图和静态图的运行时。
