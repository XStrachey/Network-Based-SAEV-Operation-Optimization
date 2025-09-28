# 图结构JSON导出工具

本工具提供了两个脚本，用于将网络优化求解结果转换为JSON格式的图结构文件。

## 脚本说明

### 1. `export_graph_to_json.py` - 完整版导出脚本

**功能**: 导出包含完整节点和边信息的详细JSON图结构

**特点**:
- 包含所有节点属性和边属性
- 包含流量分配信息
- 包含求解结果摘要
- 包含按边类型的统计信息

**使用方法**:
```bash
# 使用默认参数
python export_graph_to_json.py

# 指定输出文件
python export_graph_to_json.py --output my_graph.json

# 指定数据目录
python export_graph_to_json.py --data-dir path/to/data --output my_graph.json

# 显示详细输出
python export_graph_to_json.py --verbose
```

### 2. `export_simple_graph.py` - 简化版导出脚本

**功能**: 导出简化的图结构，专注于核心信息

**特点**:
- 只包含关键节点属性（zone, time, soc, station）
- 简化的边信息（source, target, type, cost, flow）
- 更小的文件大小
- 更快的处理速度

**使用方法**:
```bash
# 使用默认参数
python export_simple_graph.py

# 指定输出文件
python export_simple_graph.py --output simple_graph.json

# 显示详细输出
python export_simple_graph.py --verbose
```

## JSON结构说明

### 完整版JSON结构
```json
{
  "metadata": {
    "description": "Network-Based SAEV Operation Optimization Graph",
    "total_nodes": 365,
    "total_edges": 983,
    "total_flow": 2160.0,
    "total_cost": -9732.0,
    "created_at": "2025-09-28T12:25:22.773231",
    "solve_summary": { ... },
    "edge_types": { ... }
  },
  "nodes": [
    {
      "id": "node_id",
      "type": "node_type",
      "attributes": { ... }
    }
  ],
  "edges": [
    {
      "id": "edge_id",
      "source": "source_node_id",
      "target": "target_node_id",
      "type": "edge_type",
      "cost": 0.0,
      "capacity": 1000000000000.0,
      "flow": 0.0,
      "attributes": { ... }
    }
  ]
}
```

### 简化版JSON结构
```json
{
  "metadata": {
    "description": "Network-Based SAEV Operation Optimization Graph (Simplified)",
    "total_nodes": 365,
    "total_edges": 983,
    "total_flow": 2160.0,
    "total_cost": -9732.0,
    "solve_status": "Optimal",
    "objective": -9732.0,
    "edge_types": { ... }
  },
  "nodes": [
    {
      "id": "node_id",
      "type": "node_type",
      "zone": 308.0,
      "soc": 100.0
    }
  ],
  "edges": [
    {
      "id": "edge_id",
      "source": "source_node_id",
      "target": "target_node_id",
      "type": "edge_type",
      "cost": 0.0,
      "flow": 0.0
    }
  ]
}
```

## 参数说明

### 通用参数
- `--output, -o`: 输出JSON文件路径（默认：graph_structure.json 或 simple_graph.json）
- `--data-dir, -d`: 数据目录路径（默认：src/pipeline/data/solver_graph）
- `--flows-dir`: 流量数据目录路径（默认：src/pipeline/outputs）
- `--verbose, -v`: 显示详细输出信息

### 数据文件要求
脚本需要以下数据文件：
- `data/solver_graph/nodes.parquet`: 节点数据
- `data/solver_graph/arcs.parquet`: 边数据
- `outputs/flows.parquet`: 流量数据（可选）
- `outputs/solve_summary.json`: 求解摘要（可选）

## 输出示例

运行脚本后，会生成包含以下信息的JSON文件：

1. **元数据**: 图的基本信息、统计数据和求解结果
2. **节点列表**: 每个节点包含ID、类型和关键属性
3. **边列表**: 每条边包含源节点、目标节点、类型、成本和流量

## 使用建议

- **完整版脚本**: 适用于需要完整图结构信息的场景，如详细分析、可视化等
- **简化版脚本**: 适用于需要快速导出核心信息的场景，如网络分析、算法输入等

## 注意事项

1. 确保数据文件存在且格式正确
2. 大型图结构可能生成较大的JSON文件
3. 建议使用简化版脚本处理大型网络
4. 输出文件使用UTF-8编码，支持中文内容

## 问题解决

### 类型显示为"unknown"的问题
如果节点或边的类型显示为"unknown"，这通常是因为：
1. **节点数据**：节点统一类型为"node"，这是正常的
2. **边数据**：脚本会自动读取`arc_type`列作为边的类型
3. **数据列名**：脚本已适配实际的parquet文件列名结构

### 数据文件结构要求
- **节点文件** (`nodes.parquet`): 需要包含 `node_id`, `zone`, `t`, `soc`, `supply` 列
- **边文件** (`arcs.parquet`): 需要包含 `arc_id`, `arc_type`, `from_node_id`, `to_node_id`, `cost`, `capacity` 列
- **流量文件** (`flows.parquet`): 需要包含 `arc_id`, `flow` 列
