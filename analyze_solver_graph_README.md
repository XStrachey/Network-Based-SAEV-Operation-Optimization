# 求解器图分析脚本

## 概述

`analyze_solver_graph.py` 是一个用于分析 `_08_build_solver_graph.py` 生成的图规模和弧类型的Python脚本。该脚本可以读取求解器图数据文件，并提供详细的图结构分析报告。

## 功能特性

- **图规模分析**: 分析节点数量、弧数量、时间窗口等基本信息
- **弧类型统计**: 统计各种弧类型的分布、成本和容量特征
- **多格式输出**: 支持控制台、JSON和Excel格式输出
- **灵活配置**: 支持自定义数据目录路径和输出选项

## 使用方法

### 基本用法

```bash
# 使用默认设置分析图数据
python analyze_solver_graph.py
```

### 命令行参数

```bash
python analyze_solver_graph.py [选项]
```

**可用选项:**

- `--data-dir DIR`: 指定数据目录路径 (默认: `src/data/solver_graph`)
- `--output-format FORMAT`: 指定输出格式 (`console`/`json`/`csv`, 默认: `console`)
- `--save-report FILE`: 保存报告到文件
- `--quiet`: 静默模式，只输出错误信息
- `--help`: 显示帮助信息

### 使用示例

```bash
# 1. 基本分析（控制台输出）
python analyze_solver_graph.py

# 2. 指定自定义数据目录
python analyze_solver_graph.py --data-dir custom/path/to/solver_graph

# 3. 生成JSON格式报告
python analyze_solver_graph.py --output-format json --save-report report.json

# 4. 生成Excel格式报告
python analyze_solver_graph.py --output-format csv --save-report report.xlsx

# 5. 静默模式生成报告
python analyze_solver_graph.py --quiet --save-report silent_report.json

# 6. 输出JSON到控制台
python analyze_solver_graph.py --output-format json
```

## 输出格式

### 控制台输出
默认格式，提供格式化的文本报告，包括：
- 基本信息（时间窗口、节点数、弧数等）
- 节点详细信息（供给分布、时间范围、SOC范围等）
- 弧详细信息（成本范围、容量范围等）
- 弧类型分布统计

### JSON输出
结构化数据格式，包含完整的分析结果，便于程序化处理。

### Excel输出
生成Excel文件，包含两个工作表：
- `Arc_Types`: 弧类型统计表
- `Basic_Info`: 基本信息表

## 输出示例

### 控制台输出示例

```
================================================================================
求解器图规模与弧类型分析报告
================================================================================

📊 基本信息
----------------------------------------
时间窗口: t0=1, H=24, t_hi=25
节点总数: 143,216
弧总数: 8,760,194
供给总量: 200.0

🔢 节点详细信息
----------------------------------------
总节点数: 143,216
正供给节点数: 61
负供给节点数: 1
零供给节点数: 143,154
时间范围: 1 - 25
SOC范围: 0 - 100
区域数量: 221

📈 弧类型分布
----------------------------------------
弧类型总数: 10
主导类型: reposition (8,146,207 条, 93.0%)

详细分布:
弧类型             数量           百分比      平均成本         平均容量        
-----------------------------------------------------------------
reposition      8,146,207    93.0    % 1.0779       1000000000000
tochg           174,162      2.0     % 1.0000       1000000000000
svc_exit        164,732      1.9     % 0.0000       1000000000000
svc_enter       122,494      1.4     % 0.0000       1000000000000
idle            87,592       1.0     % 0.0000       1000000000000
...
```

## 依赖要求

- Python 3.7+
- pandas
- numpy
- openpyxl (用于Excel输出)

## 文件结构要求

脚本期望在指定目录中找到以下文件：
- `meta.json`: 元数据文件
- `nodes.parquet`: 节点数据文件
- `arcs.parquet`: 弧数据文件

## 错误处理

脚本包含完善的错误处理机制：
- 文件不存在检查
- 数据格式验证
- JSON序列化兼容性处理
- 详细的错误信息输出

## 注意事项

1. 确保数据文件存在且格式正确
2. Excel输出需要安装 `openpyxl` 库
3. 大型数据集可能需要较长的处理时间
4. JSON输出会自动处理numpy数据类型转换
