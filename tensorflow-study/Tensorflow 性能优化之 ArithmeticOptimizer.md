## Tensorflow 性能优化之ArithmeticOptimizer

### ArithmeticOptimizer的主要作用：

* 主要减少模型运行时的算术复杂度，提高运行性能

### ArithmeticOptimizer的功能：

1.  操作

| Optimizer stages（Pattern） | 功能介绍 |
| :------------------------------------- | -------- |
|combine_add_to_addn                    | 将Add/AddN 重写成AddN算子。如果Add输入的shape不一样，则会使用最少广播原则，减少广播的次数，将相同shape的Add节点替换成AddN，然后添加Add的节点。 |
|convert_sqrt_div_to_rsqrt_mul          | Div(x, Sqrt(y)) => Mul(x, Rsqrt(y))， |
|fold_conjugate_into_transpose          | 将共轭操作fold 到transpose中，因为有ConjugateTranspose Op。  |
|fold_multiply_into_conv                | 将一个scalar的乘数fold到卷积Conv中，权重weight = weight * scalar，例如： Conv2D(input * scalar, weight) => Conv2D(input, scalar * weight) 。 可以跨越仅对数据重新排序的节点（例如，reshape和transpose）。 如果scalar和weight 都为常量可以通过constant-folding 折叠成一个常量。 |
|fold_transpose_into_matmul             | 将Transpose  折叠入到矩阵乘法matmul中，因为MatMul有transpose的属性。 |
|fuse_squared_diff                      | 重写：Square(Sub(x, y)) => Identity(SquaredDifference(x, y)) |
|hoist_common_factor_out_of_aggregation | 使用乘除法加法的分布特性以及前者的可交换性，从所有输入均为Mul / Div节点的聚合节点中提取公共因子/分母。比如：                                                                       AddN(Mul(x, y1), Mul(y2, x), ... Mul(x, yn)) =>    Mul(x, AddN(y1, y2, ... yn)) AddN(Div(y1, x), Div(y2, x), ... Div(yn, x)) =>  Div(AddN(y1, y2, ... yn), x) |
|hoist_cwise_unary_chains               | 可将输入的一元unary运算的通用前缀提升到concat外面，或者提升到split里面。比如：Concat([Exp(Sin(x)), Exp(Sin(y)), Exp(Sin(z))]) => Exp(Sin(Concat([x, y, z]))).  或者：[Exp(Sin(y)) for y in Split(x)]  => [y for y in Split(Exp(Sin(x))] |
|minimize_broadcasts                    | 对二元相关运算进行重新排序，以最大程度地减少广播次数和临时张量的大小。例如：Add(Mul(x,Y), Mul(X,y)) => Add(Mul(x,y), Mul(X,Y))  x，y为标量， X，Y为矩阵 |
|optimize_max_or_min_of_monotonic       | 检查逐元素单调函数的Min/Max 缩减，例如Sqrt，Sigmoid，Tanh等。Max(Sqrt(x)) => Sqrt(Max(x)) . 如果relu 等激活函数前面是Biasadd 或者FusedBatchNorm，不做操作。 |
|remove_idempotent                      | 删除idempotent算子，比如CheckNumerics， Snapshot和DeepCopy等算子。 |
|remove_identity_transpose              | 移除逆转置节点。 |
|remove_involution                      | involution 是一个逐元素函数 f(x) ，他有自身的逆运算，我们可以从图中删除involution的两个实例。因为他们两个互逆函数运算是输入本身。 |
|remove_logical_not                     | 移除LogicalNot运算，根据输入，重写node的OPs，比如：LogicalNot(IsEqual(x))  =>       NotEqual(x) |
|remove_negation                        | 移除负运算，例如：a - (-b) = a + b or a + (-b) = a - b |
|remove_redundant_bitcast               | 移除冗余的Bitcasts. Bitcast 它的 source type 和 destination type相同的会被移除，重写Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2) |
|remove_redundant_cast                  | 移除冗余的cast，Cast 它的 source type 和 destination type相同的会被移除 |
|remove_redundant_reshape               | 移除冗余的Reshape和BroadcastTo节点。 |
|reorder_cast_like_and_value_preserving | 重新排序Cast使得转置处理的数据量较小。如果：sizeof(tensor.type) < sizeof(dst_type) Op(Cast(tensor, dst_type)) => Cast(Op(tensor), dst_type)  如果sizeof(tensor.type) > sizeof(dst_type) ， Cast(Op(tensor), dst_type) => Op(Cast(tensor, dst_type)) |
|replace_mul_with_square                | 替代Mul 算子为Square算子 |
|simplify_aggregation                   | 简化聚合（例如AddN）节点，比如AddN(x, x, x, ... ,x) => Mul(Const(N), x)) |
|convert_pow                            | 将Pow 算子替换成square、Sqrt、Rsqrt和Reciprocal等算子。比如pow(x,3) => mul(x, inner_square(x)) |
|convert_log1p                          | 重写 log(1+x),  log(1 + x) => log1p(x) |
|convert_log_softmax                    | Log(Softmax(x)) => LogSoftmax(x) |
|convert_expm1                          | exp(x) - 1 => expm1(x) |
|unary_ops_composition                  | 将类型和形状为一元运算的链替换为“ _UnaryOpsComposition”节点。 |
|remove_stack_slice_same_axis           | 替代格式：x = stack((a_0, a_1, ..., a_{n-1}), axis=k)[:,...,i,...] => a_i |
|simplify_embedding_lookup              | 在稀疏lookup操作期间消除不必要的copy。 |
|remove_cast_into_segment_reduction     | 消除了稀疏分段缩减操作之前不必要的强制转换。 |


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
 FindContractionWithBiasAndActivation(
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
    std::vector<>* invalidated_nodes, std::vector<>* nodes_to_delete) {
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
  (*nodes_to_delete)[matched.contraction] 
  (*nodes_to_delete)[matched.bias_add] 
  (*invalidated_nodes)[matched.activation] 

  return Status::OK();
}
```

这就完成了Conv2D+BiasAdd+Activation 的fusion 操作。

### 下图为相应的fusion 前后的图

如果Add输入的shape 一样，那么就会重写成一个AddN算子。

![same shape for AddN](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Arimetic_add_1.png)

对于输入不同的shape,会将相同shape的Add算子重写成AddN，然后其他的用AddV2/Add替代。

```c++
  //  [a, x], [b, y], [c, z] - 分别拥有相同的shape
  //
  //         +                              +
  //      /     \                       /       \
  //     +       +                     +       AddN(c, z)
  //    / \     / \                 /     \
  //   +   c   x   + -->    AddN(a, x)  AddN(b, y)
  //  / \         / \
  // a   b       y   z
```

![add](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Arimetic_add.png)

下面我们测试了对于hoist_common_factor_out_of_aggregation, 如下图所示，会将相同的因子提出来。

```c++
      // We expect the following rewrite(s) to occur:
      //
      //        Add                 Mul
      //      /    \               /   \
      //    Mul    Mul       ->   x    Add
      //    / \    / \                 / \
      //   x  y1  y2  x              y1   y2
```

![Hoist for Mul](https://github.com/gyshi/blog/blob/master/tensorflow-study/img/Arimetic_Hoist_Mul.png)