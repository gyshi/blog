## Tensorflow 性能优化之Remapper

### Remapper的主要功能：

* 主要用于推理阶段的算子融合（operator  fusion）操作。

### Remapper的提供的融合操作：

1.  共用的fusion 操作

| pattern                                            | fusion ops                                              |
| -------------------------------------------------- | ------------------------------------------------------- |
| {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd            | _Fused{Conv2D,DepthwiseConv2dNative,MatMul}             |
| {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd+Activation | _Fused{Conv2D,DepthwiseConv2dNative,MatMul}             |
| FusedBatchNorm + SideInput + Activation            | _FusedBatchNormEx                                       |
| FusedBatchNorm                                     | 将FusedBatchNorm 拆分为Add Mul Rsqrt Sub premitive 算子 |

2. MKL CPU 优化的 Fusion 操作：

| pattern                       | fusion ops   |
| ----------------------------- | ------------ |
| Conv2D+BiasAdd+Add+Activation | _FusedConv2D |
| Conv2D+BiasAdd+Add            | _FusedConv2D |

3. MKL 不支持但是eigen 和GPU 支持的融合操作

| pattern                          | fusion ops           |
| -------------------------------- | -------------------- |
| Conv2D+Squeeze+BiasAdd           | _FusedConv2D+Squeeze |
| Conv2D+FusedBatchNorm            | _FusedConv2D         |
| Conv2D+FusedBatchNorm+Activation | _FusedConv2D         |

### Remapper 算法优化流程：

> 1.  将Graph 进行拓扑排序。
> 2. 从当前的node 进行查询， 寻找相应的匹配模式(例如：Conv2D+BiasAdd+Relu)，如果寻找到相应的匹配，将匹配的node的相关信息记录下来(在Contraction开头的结构体中)。
> 3. 通过相应的匹配node的信息，建立新的fusion后的node(例如：_FusedConv2D)，这里最新的node的名字和相应的刚开始匹配的node的名字一致( _FusedConv2D 和 Relu的名字一样)。
> 4.  删除旧的node(例如：Conv2D, BiasAdd, Relu)。
> 5.  最后获得优化后的graph

下面我主要从Conv2D+BiasAdd+Activation 的fusion操作深入探讨remapper的具体操作：

1. 下面是一个提取相应匹配信息的结构体，主要记录匹配的Conv2D, BiasAdd 和Activation的node的索引。

```c++
// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(int contraction, int bias_add,
                                      int activation)
      : contraction(contraction), bias_add(bias_add), activation(activation) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
};
```

2. FindContractionWithBiasAndActivation这个函数主要用于寻找graph中匹配的节点

```c++
bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  //从activation节点开始遍历
  if (HasControlFaninOrFanout(*node_view)) return false;
  const auto* node_def = node_view->node();
  if (!IsSupportedActivation(*node_def)) return false;

  // 然后寻找activation节点的输入，向上开始遍历寻找匹配的模式。
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* bias_add_node_view = regular_fanin_0.node_view();
  const auto* bias_add_node_def = bias_add_node_view->node();

  ContractionWithBiasAdd base;
  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), &base,
                               /*check_device_compatible=*/false) ||
      !HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      !HaveSameDataType(node_def, bias_add_node_def) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;

  // 得到contraction节点
  const auto* contraction_node_view =
      bias_add_node_view->GetRegularFanin(0).node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Currently, only matmul + bias + tanh is enable
  if (!IsMatMul(*contraction_node_def) && IsTanh(*node_def)) return false;

  // Currently, only conv + bias + leakyrelu is enabled
  if (!IsConv2D(*contraction_node_def) && IsLeakyRelu(*node_def)) return false;

  // 对匹配的节点进行数据类型和Device的检查
  const ContractionWithBiasAddAndActivation pattern{base.contraction,
                                                    base.bias_add, node_index};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // 获得 {Conv2D, MatMul}+BiasAdd+Activation 匹配.
  *matched = pattern;

  return true;
}
```

3. 将fusion的节点加入，替换掉原来的匹配模式中的节点。

```c++
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);

  VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd and "
          << activation.op() << ":"
          << " activation=" << activation.name()
          << " bias_add=" << bias_add.name()
          << " contraction=" << contraction.name();
  // 建立fusion 的op， 然后将匹配模式中的输入添加到新的fusion的op中。biasadd有另外的输入bias，
  // 而activation 没有额外的输入。
  NodeDef fused_op;
  fused_op.set_name(activation.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(bias_add.input(1));     // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
    // leaky relu has a special attribute alpha
    CopyConv2DAttributes(contraction, &fused_op, &activation);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_op);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_op);
  }
  // 这一步是将biasadd 和activation 作为fused_op的属性
  SetFusedOpAttributes(&fused_op, {"BiasAdd", activation.op()});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  // 将相应旧的node 添加到删除的集合中。
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return Status::OK();
}
```

这就完成了Conv2D+BiasAdd+Activation 的fusion 操作。

### 下图为相应的fusion 前后的图

![](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Remapper_0.png)

![](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Remapper_1.png)

![](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Remapper_2.png)

![](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Remapper_3.png)


### 结论：

>   虽然fusion 能提高inference的性能，但是做fusion 很多时候还是需要手写pattern操作。目前fusion 也只是覆盖了常见的一些算子的pattern，对于未来网络模型的复杂性和多样性，手写pattern 会带来更多的工作量。

