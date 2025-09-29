# SAEV运营优化 - 场景配置系统使用说明

## 概述

本项目采用基于JSON配置的架构，支持通过不同的配置文件运行不同的定价策略场景。这种设计使得配置更加灵活，易于扩展和维护。

## 架构特点

### 1. 基于文件路径的配置
- 每个场景通过独立的JSON配置文件定义
- 定价策略通过不同的`coeff_schedule.csv`文件实现
- 配置参数通过文件路径指定，避免硬编码

### 2. 统一的配置管理
- 所有配置都通过`network_config.py`统一管理
- 支持从JSON文件动态加载配置
- 保持向后兼容性

## 使用方法

### 1. 列出所有可用场景
```bash
python src/pipeline/run_scenario.py --list
```

### 2. 运行特定场景
```bash
# 运行FCFS基线场景
python src/pipeline/run_scenario.py --scenario fcfs

# 运行TOU高峰定价场景
python src/pipeline/run_scenario.py --scenario tou_peak

# 运行TOU非高峰定价场景
python src/pipeline/run_scenario.py --scenario tou_off_peak
```

### 3. 查看详细输出
```bash
python src/pipeline/run_scenario.py --scenario fcfs --verbose
```

## 场景配置

### 现有场景

#### 1. FCFS基线 (`fcfs.json`)
- **描述**: 先到先服务固定定价基线
- **定价**: 无价格差异 (tou_price = 0.0)
- **配置文件**: `data/coeff_schedule_fcfs.csv`

#### 2. TOU高峰定价 (`tou_peak.json`)
- **描述**: 分时电价 - 高峰定价
- **定价**: 高峰时段(8-11, 14-19)价格为0.8，其他时段为0.1
- **配置文件**: `data/coeff_schedule_tou_peak.csv`

#### 3. TOU非高峰定价 (`tou_off_peak.json`)
- **描述**: 分时电价 - 非高峰定价
- **定价**: 所有时段统一价格为0.1
- **配置文件**: `data/coeff_schedule_tou_off_peak.csv`

## 配置文件结构

### JSON配置文件格式
```json
{
  "name": "场景名称",
  "description": "场景描述",
  "paths": {
    "zones": "data/zones.csv",
    "stations": "data/stations.csv",
    "od_matrix": "data/od_matrix.parquet",
    "base_ij": "data/base_ij.parquet",
    "base_i2k": "data/base_i2k.parquet",
    "coeff_schedule": "data/coeff_schedule_xxx.csv",
    "coeff_energy": "data/coeff_energy.csv",
    "fleet_init": "data/fleet_init.csv"
  },
  "network": {
    "t0": 1,
    "H": 8,
    "dt_minutes": 15
  },
  "charging": {
    "queue_relax_factor": 1.2,
    "min_charge_step": 20,
    "default_plugs_per_station": 1
  },
  "costs": {
    "vot": 1.0,
    "gamma_rep": 1.0,
    "beta_toCHG": 1.0,
    "beta_chg": 1.0,
    "service_weight_default": 15.7,
    "idle_opportunity_cost": 10.0,
    "gamma_reposition_reward": 0.2,
    "beta_chg_reward": 0.02
  },
  "solver": {
    "solver_name": "glpk",
    "time_limit_sec": 3600,
    "verbose": true
  },
  "fleet": {
    "total_fleet_size": 200,
    "initial_soc_level": 60
  },
  "output": {
    "dir": "outputs_xxx",
    "include_visualization": true
  }
}
```

### coeff_schedule CSV文件格式
```csv
t,gamma_rep_p,beta_chg_p1,beta_chg_p2,tou_price
1,1,1,1,0.0
2,1,1,1,0.0
...
```

## 创建新场景

### 1. 创建coeff_schedule文件
```python
import pandas as pd

# 创建新的定价策略
def create_tou_schedule(output_path, peak_price, off_peak_price):
    # 读取基础schedule文件
    base_schedule = pd.read_csv("data/coeff_schedule.csv")
    
    # 定义高峰时段
    peak_hours = [8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    
    # 添加TOU价格列
    base_schedule['tou_price'] = base_schedule['t'].apply(
        lambda t: peak_price if t in peak_hours else off_peak_price
    )
    
    # 保存到新文件
    base_schedule.to_csv(output_path, index=False)

# 创建新场景的coeff_schedule文件
create_tou_schedule("data/coeff_schedule_my_scenario.csv", 1.0, 0.2)
```

### 2. 创建JSON配置文件
```json
{
  "name": "My Custom Scenario",
  "description": "自定义定价策略",
  "paths": {
    "coeff_schedule": "data/coeff_schedule_my_scenario.csv"
  },
  "output": {
    "dir": "outputs_my_scenario"
  }
}
```

### 3. 运行新场景
```bash
python src/pipeline/run_scenario.py --scenario my_scenario
```

## 输出结果

每个场景运行后会生成以下文件：

### 1. 求解结果
- `outputs/flows.parquet`: 流量数据
- `outputs/solve_summary.json`: 求解摘要

### 2. 可视化结果
- `viz/arc_flows_timeline_corrected.html`: 时间线图表
- `viz/arc_flows_combined_corrected.html`: 组合图表

### 3. 求解摘要示例
```json
{
  "status": "Optimal",
  "objective": 5229.0,
  "total_cost": 5229.0,
  "total_flow": 1940.0,
  "nodes": 122,
  "arcs": 212,
  "by_type": {
    "from_source": {"flow": 200.0, "cost": 0.0, "arcs": 2},
    "idle": {"flow": 840.0, "cost": 8400.0, "arcs": 74},
    "reposition": {"flow": 70.0, "cost": 126.0, "arcs": 2},
    "svc_enter": {"flow": 210.0, "cost": 0.0, "arcs": 28},
    "svc_exit": {"flow": 210.0, "cost": 0.0, "arcs": 72},
    "svc_gate": {"flow": 210.0, "cost": -3297.0, "arcs": 12},
    "to_sink": {"flow": 200.0, "cost": 0.0, "arcs": 22}
  }
}
```

## 技术实现

### 1. 配置加载器
- `ConfigLoader.load_from_json()`: 从JSON文件加载配置
- `get_network_config(scenario)`: 获取指定场景的配置

### 2. 场景运行器
- `run_scenario.py`: 主运行脚本
- 支持场景列表、运行、详细输出等功能

### 3. Pipeline集成
- `_01_build_solver_graph.py`: 支持`--scenario`参数
- 自动从JSON配置加载参数

## 优势

1. **灵活性**: 通过文件路径指定不同配置，易于扩展
2. **可维护性**: JSON配置清晰易读，便于修改
3. **可扩展性**: 轻松添加新场景，无需修改代码
4. **统一性**: 所有配置通过统一接口管理
5. **向后兼容**: 保持原有配置系统兼容性

## 注意事项

1. 确保虚拟环境已激活
2. 检查数据文件路径是否正确
3. 新场景的coeff_schedule文件必须包含所有时间步
4. JSON配置文件必须包含所有必需字段
5. 输出目录会自动创建，但需要确保有写入权限
