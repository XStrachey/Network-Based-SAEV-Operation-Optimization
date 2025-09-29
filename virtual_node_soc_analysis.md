# 虚节点电量显示与充电容量约束分析

## 问题总结

### 1. 虚节点电量显示问题

**问题描述**：
- 虚节点（负ID节点）在原始数据中只有`supply`属性，缺少`t`、`zone`、`soc`等时空和电量信息
- 用户无法直观了解虚节点上车辆的电量状态

**解决方案**：
- 修改了`viz_graph.py`，从边数据中提取虚节点的SOC信息
- 在虚节点的tooltip中显示：
  - `soc from`: 起始SOC值
  - `soc to`: 目标SOC值
- 这些信息来自与虚节点相连的边的`l`和`l_to`字段

### 2. 充电容量约束问题

**问题描述**：
- 不同SOC级别的车辆共用同一个`chg_occ`弧的容量约束
- 从数据中可以看到：`chg_occ`弧的`l=60.0, l_to=60.0`，意味着所有SOC级别的车辆都使用同一个充电容量
- 这可能导致充电资源分配不合理

**技术细节**：
```python
# 在charging_arc.py中，chg_occ弧是这样创建的：
chg_occ_arc = ArcMetadata(
    arc_type="chg_occ",
    from_node_id=q_in_id,  # q_in(k,p)
    to_node_id=q_out_id,   # q_out(k,p)
    # 容量约束在站点k和时间p维度上，没有区分SOC级别
    cap_hint=station_capacity
)
```

**影响**：
- 不同SOC级别的车辆竞争同一个充电容量
- 可能导致低SOC车辆无法及时充电
- 充电效率可能不是最优的

## 改进建议

### 1. 短期改进（可视化层面）
- ✅ 已完成：在虚节点tooltip中显示SOC信息
- ✅ 已完成：在充电相关边的tooltip中显示SOC信息
- ✅ 已完成：特别标注`chg_occ`的容量共享问题

### 2. 长期改进（模型层面）
- 考虑按SOC级别创建不同的充电容量约束
- 或者按SOC范围分组创建充电容量约束
- 这样可以更精确地建模充电资源分配

## 文件修改

### viz_graph.py 修改内容：
1. 添加了`virtual_soc_info`字典来收集虚节点的SOC信息
2. 在虚节点tooltip中添加SOC信息显示
3. 在充电相关边的tooltip中添加SOC信息和容量约束警告

### 生成的新文件：
- `graph_with_soc_info.html`: 包含SOC信息的增强版可视化

## 使用方法

```bash
# 生成包含SOC信息的可视化
python viz_graph.py -i simple_graph.json -o graph_with_soc_info.html

# 在浏览器中打开查看
open graph_with_soc_info.html
```

## 验证结果

通过修改后的可视化，用户可以：
1. 点击虚节点查看相关的SOC信息
2. 点击充电相关边查看SOC变化和容量约束
3. 识别`chg_occ`边的容量共享问题

这为后续的模型优化提供了重要的可视化支持。
