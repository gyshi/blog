## Tensorflow 性能优化之BFloat16 源码分析

### BFloat16 数据格式

  bfloat16数据格式为[1:8:7]，它具有一个符号位，八个指数位，七个尾数位以及一个隐式尾数位。相比之下，标准的16位浮点（fp16）格式为[1:5:10]。fp16格式只有5个指数位,由于这些特性，bfloat16的动态范围比fp16大。bfloat16范围对于诸如梯度之类的事情很有用，因为这些梯度可能超出fp16的动态范围，因此需要进行损耗缩放。bfloat16可以直接表示这样的渐变。bfloat16的动态范围大于fp16的动态范围,使用bfloat16可以减少内存中的数据大小，并允许较大的模型容纳相同数量的内存。

  某些操作受内存带宽限制，这意味着内存带宽决定了此类操作所花费的时间。以bfloat16格式存储内存带宽受限操作的输入和输出可减少必须传输的数据量，从而提高了操作速度。格式是32位[IEEE 754单精度浮点格式](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)（binary32）的截断（16位）版本，旨在[加速](https://en.wikipedia.org/wiki/Hardware_acceleration)[机器学习](https://en.wikipedia.org/wiki/Machine_learning)和[近传感器计算](https://en.wikipedia.org/wiki/Intelligent_sensor)的bfloat16格式，作为一个截短的[IEEE 754单精度](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) 32位浮点，允许快速[转换](https://en.wikipedia.org/wiki/Type_conversion)到和从IEEE 754单精度32位浮点。

![bfloat16](C:\Users\shiguang\shiguangyong blog\日常博客总结\img\bfloat16.png)

> Tensorflow  [bfloat16](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/bfloat16.h)代码实现:
>
> ```C++
> // Conversion routines between an array of float and bfloat16 of
> // "size".
> void FloatToBFloat16(const float* src, bfloat16* dst, int64 size);
> void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size);
> ```
>
> 下面是[源码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/bfloat16.cc)：
>
> ```c++
> void FloatToBFloat16(const float* src, bfloat16* dst, int64 size) {
>   const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
>   uint16_t* q = reinterpret_cast<uint16_t*>(dst);
> #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
>   for (; size != 0; p += 2, q++, size--) {
>     *q = p[0];
>   }
> #else
>   for (; size != 0; p += 2, q++, size--) {
>     *q = p[1];
>   }
> #endif
> }
> void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size) {
>   const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
>   uint16_t* q = reinterpret_cast<uint16_t*>(dst);
> #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
>   for (; size != 0; p++, q += 2, size--) {
>     q[0] = *p;
>     q[1] = 0;
>   }
> #else
>   for (; size != 0; p++, q += 2, size--) {
>     q[0] = 0;
>     q[1] = *p;
>   }
> #endif
> ```
>
>   这里实现了大端和小端的数据转换。float32 转到bfloat16 只是做了简单的截断，bfloat16 转换到float32，在低16位补0。

###  BFloat16 的算法流程

> 这里我是用的intel的BFloat16 作为案例。
>
> intel 的blfoat16 主要支持3rd Generation Intel® Xeon® Scalable Processors，需要重新编译tensorflow ，必须有
>
> `--config=mkl 和--copt=-DENABLE_INTEL_MKL_BFLOAT16`

tensrflow的F16(bfloat16 and fp16) 有四个list ：这里因为tensorflow 不同的版本对这些list的名字命名不太一样（可能种族颜色问题）.

| list 名字             | 描述                                                         |
| --------------------- | ------------------------------------------------------------ |
| AllowList(White List) | numerically-safe、performance-critical、can run in BF16.     |
| InferList(Gray List)  | numerically-safe、may be made unsafe by an upstream denylist ops op、can run in BF16. |
| DenyList(Black List)  | numerically-dangerous  effects may also be observed in downstream nodes |
| ClearList             | Do not have numerically-significant effects、Can run in BF16 |

算法流程（包含在Optimizer()函数里：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/auto_mixed_precision.cc#L1241）主要分为下面几个步骤，按顺序执行：

> 1. 根据TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ __*list name*__ _ ADD 和    TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ __*list name*__ _ADD 得到AllowList、DenyList、InferList、ClearList。这里list_name 代表的是ALLOWLIST、INFERLIST、DENYLIST和CLEARLIST。
> 2. 进行预处理将FusedBatchNorm 转换为FusedBatchNormV2， 同时建立graph的Node和属性类型TypeAttr的映射，可以通过映射关系查找node的输入输出端口和相应的TyypeAttr属性。
> 3. 构造graph的type attr的拓扑视图，如果nodeA 具有TypeAttrID ‘T’属性的输出 A:0 ， node B有TypeAttrID  ‘U’ 的输入B:0 。输入(B:0) -> (A:0)， 通过处理会变成（B:U） -> (A:T)。这一步主要用于根据node的attr和name 获取node、node再graph中的index 和fanin/fanout。
> 4. 将在AllowList中ops的node 加入到allow_set 中。
> 5. 将denylist中的node 和在denylist的node到deny/infer node路径上属于inferlist、clearlist和denylist的node加入到deny_set。因为inferlist的node会受到上游的denylist中node的影响。
> 6. 强制Tensor list的node在相同的set中。
> 7. 将allow ops之间的clear和infer node加入到allow_set中，然后处理剩余的clear ops，并将符合条件的加入到allot_set中。
> 8. 强制Tensor list的node在相同的set中，进行循环边（loop edges）的匹配，同时对已经存在graph中的Cast node 进行处理，对于符合条件的加入到allow_set中，减少多余的cast的插入。
> 9. 最后根据allow_set，将type_attr 转换成 DT_HALF 或者 DT_BFLOAT16， 然后对转换后type attribute 的node 进行遍历，对于前后type attr不一致的插入cast 节点。

### 核心源码解析

在这里，核心的代码主要分为struct TypeAttrId, class NodeTypeAttrMap ,struct NodeTypeId,class GraphTypeTopologyView 和void DfsTypeTraversal。

```c++
struct TypeAttrId{
  static constexpr int kSingleType = -1;

  explicit TypeAttrId(const string& _attr_name, int _type_index = kSingleType)
      : attr_name(_attr_name),
        type_index(_type_index),
        fixed_type(DT_INVALID) {}

  explicit TypeAttrId(DataType _fixed_type)
      : attr_name(), type_index(kSingleType), fixed_type(_fixed_type) {}

  string attr_name;
  // If attr_name is a list(type), this is the index into the list. Otherwise
  // this is kSingleType.
  int type_index;
  DataType fixed_type;
}
```

TypeAttrId主要表示一个node的类型属性，如果属性名是一个list，type_index存储的是它在列表的索引。

```c++
// A utility class to lookup node type attributes and type attribute <->
// input/output port mappings.
class NodeTypeAttrMap {
 public:
  NodeTypeAttrMap() {}

  explicit NodeTypeAttrMap(const GraphDef& graph) { TF_CHECK_OK(Init(graph)); }
   
  Status Init(const GraphDef& graph)
  ....
  // Returns the set of all type attributes in the given node.
  absl::flat_hash_set<TypeAttrId> GetTypeAttrs(const NodeDef& node) const 
  const absl::flat_hash_set<int>& GetInputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const
      
  const absl::flat_hash_set<int>& GetOutputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const

  TypeAttrId GetInputTypeAttr(const NodeDef& node, int port) const 

  TypeAttrId GetOutputTypeAttr(const NodeDef& node, int port) const 

 private:
   Status AddNode(const NodeDef& node) 
  // WARN: `graph_` must outlive this object (node pointers must remain valid).
  const GraphDef* graph_ = nullptr;  // do not own
  std::unique_ptr<FunctionLibraryDefinition> function_library_;
  typedef absl::flat_hash_set<int> IntSet;
  // Maps a type attr id -> (input port set, output port set)
  typedef absl::flat_hash_map<TypeAttrId, std::pair<IntSet, IntSet>> Type2IOMap;
  // Maps a node -> type attr mapping
  absl::flat_hash_map<const NodeDef*, Type2IOMap> type2io_;
  // Maps a port -> type attr id
  typedef std::vector<TypeAttrId> TypeAttrIdVec;
  // Maps a node -> (input port mapping, output port mapping)
  absl::flat_hash_map<const NodeDef*, std::pair<TypeAttrIdVec, TypeAttrIdVec>>
      io2type_;
};
```

NodeTypeAttrMap用于查找node的相应端口的属性和根据node和TypeAttrId查找输入/输出端口。它提供了type2io_  和 io2type_  存储相应的映射关系。type2io_  存储的是一个node的相应TypeAttrId的输入/输出端口id。io2type_ 存储的是一个node的输入或者输出端口的TypeAttrId。type2io_  和io2type_ 都是通过AddNode函数处理Graph中Node。下面是AddNode的核心代码，主要是下面两个for 循环，将type_attr 和input/output port 的映射关系整理出来。

```c++
 Status AddNode(const NodeDef& node) {
    const OpDef* op_def_ptr = nullptr;
    TF_RETURN_IF_ERROR(function_library_->LookUpOpDef(node.op(), &op_def_ptr));
    const OpDef& op_def = *op_def_ptr;
    auto& type2io_entry = type2io_[&node];
    auto& io2type_entry = io2type_[&node];
    auto input_arg_inds = InputPortArgDefIndexes(node, op_def);
    if (NonControlInputs(node).size() != input_arg_inds.size()) {
      return errors::InvalidArgument(
          "Expected ", node.op(), " node ", node.name(), " to have ",
          input_arg_inds.size(), " non-control input(s), but got ",
          node.input_size());
    }
    io2type_entry.first.reserve(input_arg_inds.size());
    for (int i = 0; i < static_cast<int>(input_arg_inds.size()); ++i) {
      const auto& arg_inds = input_arg_inds[i];
      const OpDef::ArgDef& arg_def = op_def.input_arg(arg_inds.first);
      TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
      if (!type_attr.attr_name.empty() &&
          !node.attr().count(type_attr.attr_name)) {
        return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                       " is not present in node ", node.name());
      }
      type2io_entry[type_attr].first.insert(i);
      io2type_entry.first.push_back(type_attr);
    }

    auto output_arg_inds = OutputPortArgDefIndexes(node, op_def);
    io2type_entry.second.reserve(output_arg_inds.size());
    for (int i = 0; i < static_cast<int>(output_arg_inds.size()); ++i) {
      const auto& arg_inds = output_arg_inds[i];
      const OpDef::ArgDef& arg_def = op_def.output_arg(arg_inds.first);
      TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
      if (!type_attr.attr_name.empty() &&
          !node.attr().count(type_attr.attr_name)) {
        return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                       " is not present in node ", node.name());
      }
      type2io_entry[type_attr].second.insert(i);
      io2type_entry.second.push_back(type_attr);
    }
    ....
    return Status::OK();
  }

```

NodeTypeId是一个struct， 存储node和相应的TypeAttrId。

```c++
struct NodeTypeId {
  NodeTypeId(const NodeDef* _node, const TypeAttrId& _type_attr)
      : node(_node), type_attr(_type_attr) {}
  ....
}
```

GraphTypeTopologyView 主要得到node的index的fanin和fanout关系，在InitializeFromGraph函数中通过NodeTypeAttrMap中的输入输出端口，找到当前node的fanin或者fanout，然后存储到fanins_ 和fanins_。

```c++
class GraphTypeTopologyView {
 public:
  GraphTypeTopologyView() = default;
  explicit GraphTypeTopologyView(bool skip_invalid_edges)
      : skip_invalid_edges_(skip_invalid_edges) {}

  Status InitializeFromGraph(const GraphDef& graph,
                             const NodeTypeAttrMap& node_type_map);

  Status AddEphemeralEdges(absl::Span<const NodeTypeIdEdge> ephemeral_edges);

  bool is_initialized() const { return graph_ != nullptr; }
  int num_nodes() const { return num_nodes_; }
  const GraphDef* graph() const { return graph_; }

  // Returns true iff the node exists in the underlying graph.
  bool HasNode(absl::string_view node_name, const TypeAttrId& type_attr) const;

  // Finds a node by name or returns `nullptr` if it's not in the graph.
  const NodeTypeId* GetNode(absl::string_view node_name,
                            const TypeAttrId& type_attr) const;
  // Returns a node corresponding to the given node index.
  const NodeTypeId* GetNode(int node_idx) const;

  // Returns a node index for the given node name, if the name exists in the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(absl::string_view node_name,
                                         const TypeAttrId& type_attr) const;
  // Returns a node index for the given node, if the node belongs to the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(const NodeTypeId& node) const;

  // Returns all the node indexes that are in the direct fanin of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 4>& GetFanin(int node_idx) const;
  // Returns all the node indexes that are in the direct fanout of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 2>& GetFanout(int node_idx) const;

 private:
  // The key type used to uniquely identify a type attribute on a node.
  struct NodeTypeKey : public std::pair<absl::string_view, TypeAttrId> 

  bool skip_invalid_edges_ = false;

  const GraphDef* graph_ = nullptr;  // do not own
  int num_nodes_ = 0;
  std::vector<NodeTypeId> node_type_attrs_;
  absl::flat_hash_map<absl::string_view, int> node_name_to_index_;
  absl::flat_hash_map<NodeTypeKey, int> node_type_name_to_index_;

  std::vector<absl::InlinedVector<int, 4>> fanins_;
  std::vector<absl::InlinedVector<int, 2>> fanouts_;

  // We need a valid reference to return from GetFanin/GetFanout if the
  // `node_idx` argument is outside of the [0, num_nodes_) range.
  absl::InlinedVector<int, 4> empty_fanin_;
  absl::InlinedVector<int, 2> empty_fanout_;
};

```

下面重要介绍DfsTypeTraversal这个函数，因为这是这个BF16的核心，后面对infer clear deny 的node处理都会用到DfsTypeTraversal。predicates和callbacks是通过外界的lambda函数进行传入，是进行dfs遍历的条件，通常用到的是predicates.enter 和 callbacks.pre_order，其他比如callbacks.on_back_edge 、 predicates.advance、callbacks.post_order传入的都是nullptr。通过predicates.enter决定传入的node是否可以进入进行后面的操作，即如果满足enter函数的条件即可对当前的node开始遍历，然后到callbacks.pre_order，这里主要决定是否要将合适的node进行处理，在BFloat16的代码中，主要功能是是否将node加入到相应的set中。

```c++
void DfsTypeTraversal(const GraphTypeTopologyView& graph_type_view,
                      const absl::Span<const NodeTypeId* const> from,
                      const TypeTraversalDirection directpredicatesion,
                      const DfsTypePredicates& predicates,
                      const DfsTypeCallbacks& callbacks) {
  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());

  for (const NodeTypeId* node : from) {
    const absl::optional<int> node_idx = graph_type_view.GetNodeIndex(*node);
    DCHECK(node_idx.has_value())
        << "Illegal start node: " << node->node->name();
    if (node_idx.has_value()) {
      stack.emplace_back(node_idx.value());
    }
  }

  absl::flat_hash_map<int, NodeState> node_state;
  while (!stack.empty()) {
    DfsStackElem w = stack.back();
    stack.pop_back();

    NodeState& state = node_state[w.node];
    if (state == NodeState::kDone) continue;

    // Skip nodes that we should not enter.
    if (predicates.enter && !predicates.enter(w.node)) {
      state = NodeState::kDone;
      continue;
    }

    // We've processed all the children of this node.
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(w.node);
      }
      continue;
    }

    // Loop detected.
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(w.src, w.node);
      }
      continue;
    }

    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.emplace_back(w.node, true, w.src);

    // Check if we can continue traversal from the current node.
    if (predicates.advance && !predicates.advance(w.node)) {
      continue;
    }
    // Now enqueue the fanin/fanout nodes.
    if (direction == TypeTraversalDirection::kFollowInputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanin : graph_type_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    }
    if (direction == TypeTraversalDirection::kFollowOutputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanout : graph_type_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}

```

下面我着重以一个例子介绍他的使用。这一个函数主要是将在ALlow之间的clear 和 infer node 加入到allow_set 中。主要分两步：

1. 第一步寻找allow ops下游的clear/infer ops，所以第一个for循环先做一个判断，如果当前节点可处理并且是allowlist ops，则进行DfsTypeTraversal的处理，否则直接跳过。在DfsTypeTraversal中TypeTraversalDirection::kFollowOutputs,代表寻找当前node的fanouts节点，也就是当前allow node的下游node，DfsTypePredicates::Enter代表的是如果是当前node（也就是当前node是allow ops）或者对于当前node的fanout节点满足下面几个条件：不在downstream_of_allow_set并且不是allow ops并且是 infer和clear ops，则加入到downstream_of_allow_set中，所以downstream_of_allow_set保存了allow ops和allow ops下游的infer/clear node。
2. 第二步是第二个for循环就是将在allow ops之间的clear/infer ops 加入到allow_set 中。kFollowInputs代表当前node的fanin也就是其上游node。这里通过upstream_of_allow_set记录遍历过的downstream_of_allow_set中的当前node的上游node，对当前node的上游node进行遍历查找，如果符合条件，则将其插入到allow set中。

```c++
void AutoMixedPrecisionImpl::AddClearAndInferToAllowIfBetweenAllow(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  // Find clear/inferlist ops that are downstream of allow ops.
  absl::flat_hash_set<int> downstream_of_allow_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || !f16_allowlist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!downstream_of_allow_set.count(idx) &&
                  !f16_allowlist_.count(item.node->op()) &&
                  !deny_set.count(idx) && ShouldProcess(*item.node) &&
                  IsFloat32(item) && SupportsF16(item) &&
                  (f16_clearlist_.count(item.node->op()) ||
                   f16_inferlist_.count(item.node->op())));
        }),
        DfsTypeCallbacks::PreOrder(
            [&](int idx) { downstream_of_allow_set.insert(idx); }));
  }
  // Set nodes that are both downstream and upstream of allow ops to allow.
  absl::flat_hash_set<int> upstream_of_allow_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || upstream_of_allow_set.count(root_idx) ||
        !f16_allowlist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowInputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!upstream_of_allow_set.count(idx) &&
                                     downstream_of_allow_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          upstream_of_allow_set.insert(idx);
          bool inserted = allow_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " ALLOW";
          }
        }));
  }
}
```

 最后一个ChangeTypeAttrsAndAddCasts函数主要是针对allow_set中的函数，通过遍历graph所有的node，然后根据node的type attr，对于在allow_set中的op，改变datatype为bfloat16或者fp16，然后根据node的name 和 type attr 找到node的fanout node， 如果当前node和fanout的node都在allow_set中，则不需要插入过多的cast，否则插入cast进行data type的转换。

```c++
Status AutoMixedPrecisionImpl::ChangeTypeAttrsAndAddCasts(
    const absl::flat_hash_set<int>& allow_set) {
  int num_nodes_changed = 0;
  int num_nonvar_casts_to_f16 = 0;
  int num_nodes_preop = graph_->node_size();
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    for (const TypeAttrId& type_attr : node_type_map_.GetTypeAttrs(*node)) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node->name(), type_attr);
	  ...
      int node_type_idx = maybe_node_type_idx.value();
      if (!IsFloat32(*graph_type_view_.GetNode(node_type_idx))) continue;
      bool src_is_allow = allow_set.count(node_type_idx);
      if (src_is_allow) {
		...
        if (!SetDataType(node, type_attr, target_dtype_)) {
          return errors::Internal("Failed to set type attribute");
        }
        ++num_nodes_changed;
      }
      for (int output_port : node_type_map_.GetOutputPorts(*node, type_attr)) {
        MutableGraphView::OutputPort src(node, output_port);
        NodeDef* added_cast_node = nullptr;
        // Note: This is copied so that edges can be modified inside the loop.
        auto fanout = graph_view_.GetFanout(src);
        for (const MutableGraphView::InputPort& dst : fanout) {
          TypeAttrId dst_type_attr =
              node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
          const absl::optional<int> maybe_dst_type_idx =
              graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
          if (!maybe_dst_type_idx.has_value()) {
            return ....
          }
          int dst_type_idx = maybe_dst_type_idx.value();
          bool dst_is_allow = allow_set.count(dst_type_idx);
          if (src_is_allow != dst_is_allow) {
            if (!added_cast_node) {
              bool to_f16 = dst_is_allow;
              ...
              added_cast_node = graph_view_.AddNode(
                  BuildCastNode(src, to_f16, src.node->device()));
              if (to_f16 && !IsConstant(*node) && !IsVariable(*node) &&
                  !NodeImplicitlyReadsNonResourceVariable(*node)) {
                ++num_nonvar_casts_to_f16;
              }
            }
            TF_RETURN_IF_ERROR(graph_view_.UpdateRegularFaninByPort(
                dst.node->name(), dst.port_id, {added_cast_node->name(), 0}));
          }
        }
      }
    }
  }
.....
}

```

### INTEL 的BFloat16的使用及实验结果：

```python
### BFloat16的使用:

​```python
# e.g.1 for session
from tensorflow.core.protobuf import rewriter_config_pb2
...
config = tf.compat.v1.ConfigProto()
config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
with tf.compat.v1.Session(config=config):
    ....
    
# e.g.2 for estimator
from tensorflow.core.protobuf import rewriter_config_pb2
def predict():
    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
    run_config = tf.estimator.RunConfig().replace(session_config=config)
    ...
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=conf.mode;_dir, config=run_config)
​```


```

环境：
> Tensorflow(tf2.4 master 源码编译) 
> CPU: Intel CPX  batchsize=128

| model / fps | Fp32    | BF16 |
| ------------- |  ------------ | ------------------------------- |
| Resnet v2 50  | 1313.33      | 2238.98                         |
| Resnet v2 101 | 709.66      | 1265.06                         |

BF16 相对于Fp32 加速比达到了1.7x ~ 1.8x

参考：

1. [借助 TensorFlow 和 Bfloat16 在第三代 Intel® Xeon® Scalable 处理器上提高 AI 性能](https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247489929&idx=1&sn=b2d3a2b38f6885691e5f5560c26406e7&chksm=fc1852c1cb6fdbd79e13bc3e639d6a8011f9b73f0ce2690d11bc5a4cfe16ae77c481bf879cde&scene=126&sessionid=1597539820&key=c3402f98b9ff36462c3563198e5b578b48bd9a3b45fe7a5b8133f17f795a597e55d6f30de02952510ce1f913c22184ef60bdf553b2d9ddb364f3db826cff17af46c89e3d2f92b8493a28cfb64cc4232a69d834bb22e3d51e2299a36b395be097a100c41faeb899d6c40983c341e968d96fc7040114b96654b93fefd2a836022c&ascene=1&uin=MTE2OTQxMDIzNg%3D%3D&devicetype=Windows+10+x64&version=62090529&lang=en&exportkey=A9jvSfAwHCvSbxA6rK5lqBs%3D&pass_ticket=NgEhg%2FIahCjh0KxmBwZnp8ipnrQszrQD8H9Gz1iFeq4WZgaZsD2puT%2FjqVKlxjNV)

2. [Using bfloat16 with TensorFlow models](https://cloud.google.com/tpu/docs/bfloat16)

3. https://github.com/tensorflow/tensorflow (commit: 930274bb8ab10379f4c76618cccc604c9fe27996)

