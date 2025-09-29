# _01_network/network_config.py
# 网络求解方案的独立配置

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

# -------------------------
# 路径（仅必需外部输入 + 输出）
# -------------------------
@dataclass
class Paths:
    zones: str = "../../data/zones.csv"                 # 列: zone, lat, lon
    stations: str = "../../data/stations.csv"           # 列: k, zone, level, lat, lon, plugs, util_factor
    od_matrix: str = "../../data/od_matrix.parquet"     # 列: t, i, j, demand
    base_ij: str = "../../data/base_ij.parquet"         # 列: i, j, base_minutes[, dist_km]
    base_i2k: str = "../../data/base_i2k.parquet"       # 列: i, k, base_minutes[, dist_km]
    coeff_schedule: str = "../../data/coeff_schedule.csv"  # 列: t, gamma_rep_p, beta_chg_p1, beta_chg_p2
    coeff_energy: str = "../../data/coeff_energy.csv"      # 列: t, de_per_km_srv, de_per_km_rep, de_per_km_tochg
    fleet_init: str = "../../data/fleet_init.csv"       # 列: zone, soc (仅用于确定初始节点位置，不用于数量分配)

# -------------------------
# 时间与 SOC 离散
# -------------------------
@dataclass
class TimeSOC:
    dt_minutes: int = 15
    start_step: int = 1
    end_step: int = 8
    window_length: int = 8           # 窗口长度 H
    overhang_steps: int = 2          # Halo：窗外可见/可到达的"缓冲步数"（建议 2~6）
    roll_step: int = 1
    soc_levels: List[int] = None

    def __post_init__(self):
        if self.soc_levels is None:
            # 0, 5, 10, ..., 100
            self.soc_levels = list(range(0, 101, 10))

# -------------------------
# 充电/队列
# -------------------------
@dataclass
class ChargingAndQueue:
    queue_relax_factor: float = 1.2      # 站容量放松系数（1=硬容量）
    min_charge_step: int = 20            # 单次最小充电增量（SOC点）
    default_plugs_per_station: int = 1   # 默认插座数

# -------------------------
# 成本与公平性（α/β 随时间由 coeff_schedule.csv 提供）
# -------------------------
@dataclass
class CostsAndEquity:
    vot: float = 1.0

    # —— 与时间表相匹配的“常数后备值”（当 coeff_schedule 缺失时使用）——
    gamma_rep: float = 1.0               # 重定位时间系数 γ_rep_p
    beta_toCHG: float = 1.0              # 去站时间系数 β_chg_p1
    beta_chg: float = 1.0                # 充电占用系数 β_chg_p2

    # —— 服务奖励权重 —— 
    service_weight_default: float = 15.7    # svc_gate 服务奖励权重，计算公式：-VOT*service_weight_default

    # —— 新目标函数：收益项系数（请保持非负）——
    gamma_reposition_reward: float = 0.2  # 重定位收益系数 γ_rep（施加在 reposition 弧，按目的地 j 与 t）
    beta_chg_reward: float = 0.02                   # 充电收益系数 α_chg（施加在 chg_occ/step 弧，按 ΔSOC）
    
    # —— idle 机会成本 ——
    idle_opportunity_cost: float = 10             # idle 弧的机会成本（每时间步）

    # zone_value 归一化
    zone_value_normalize: str = "per_t_sum"  # 选项: "none", "per_t_sum", "per_t_max", "global_max", "window_sum"
    zone_value_eps: float = 1e-9             # 防 0 除误差

# -------------------------
# 期末SOC约束
# -------------------------
@dataclass
class EndSOCConstraints:
    end_soc_min: Optional[int] = None         # 硬下界（0-100，None表示不启用）
    end_soc_penalty_per_pct: float = 0.1      # 软惩罚（每缺1% SOC的成本）
    end_soc_total_min_ratio: float = 1.0      # 期末总能量比阈值（与期初总能量相比）

# -------------------------
# 剪枝规则（规模控制）
# -------------------------
@dataclass
class PruningRules:
    # —— 你已有的字段（原样保留）——
    max_reposition_tt: float = 45.0      # 若 τ_rep > 45 min，剪枝
    min_soc_for_reposition: int = 20     # SOC% 低于阈值不允许重定位
    reposition_nearest_zone_n: int = 16
    charge_nearest_station_n: int = 16
    
    # —— 需求驱动的重定位弧生成参数 ——
    max_reposition_pairs_per_zone: int = 3
    high_demand_threshold: int = 25
    reposition_demand_ratio: float = 0.3          # 重定位需求相对于服务需求的比例
    min_reposition_demand: float = 0.1            # 最小重定位需求阈值
    reposition_imbalance_threshold: float = 1.0   # 供需不平衡阈值

# -------------------------
# 求解器
# -------------------------
@dataclass
class SolverConfig:
    solver_name: str = "glpk"
    time_limit_sec: int = 3600           # 默认延长到 30min，更符合你现在的规模
    mip_gap: float = 0.01
    threads: int = 0
    verbose: bool = True
    RELAX_BINARIES: bool = True          # 仅控制是否放松二元为 [0,1]
    use_nopresol: bool = True            # 若上层支持，传递给 GLPK 以便超时仍能导出可行解

# -------------------------
# 模型结构开关
# -------------------------
@dataclass
class ModelFlags:
    # —— 新增：06 成本组件开关（便于 A/B 测试）——
    enable_service_reward: bool = True       # svc_gate 奖励（等价未满足惩罚）
    enable_reposition_reward: bool = True    # γ_rep * zone_value 的重定位收益
    enable_charging_reward: bool = True      # α_chg * ΔSOC 的充电收益

@dataclass
class ArcTypeControl:
    """弧类型生成控制配置"""
    # 弧类型开关
    enable_idle: bool = True         # 启用idle弧生成
    enable_service: bool = False      # 启用service弧生成
    enable_reposition: bool = True   # 启用reposition弧生成
    enable_charging: bool = False     # 启用charging弧生成
    
    # 弧类型优先级（用于调试时按顺序生成）
    generation_order: List[str] = None
    
    def __post_init__(self):
        if self.generation_order is None:
            self.generation_order = ["idle", "service", "reposition", "charging"]
    
    def get_enabled_types(self) -> List[str]:
        """获取启用的弧类型列表"""
        enabled = []
        if self.enable_idle:
            enabled.append("idle")
        if self.enable_service:
            enabled.append("service")
        if self.enable_reposition:
            enabled.append("reposition")
        if self.enable_charging:
            enabled.append("charging")
        return enabled
    
    def is_type_enabled(self, arc_type: str) -> bool:
        """检查特定弧类型是否启用"""
        return getattr(self, f"enable_{arc_type}", False)

@dataclass
class EnergyRates:
    de_per_km_srv: float = 1
    de_per_km_rep: float = 1
    de_per_km_tochg: float = 1

@dataclass
class BasicConfig:
    avg_speed_kmh: float = 80.0

# -------------------------
# 车队配置
# -------------------------
@dataclass
class FleetConfig:
    """车队配置"""
    total_fleet_size: int = 200              # 总车队规模
    initial_soc_level: int = 60             # 初始SOC水平（百分比）
    # 注意：fleet_init.csv 中的 count 列将被忽略，仅使用 zone, soc 列来确定初始节点位置

# R1PruneConfig 已移除 - 不再需要生成后裁剪，改为需求驱动的生成端控制

# -------------------------
# 全局 Config
# -------------------------
@dataclass
class NetworkConfig:
    # 使用 default_factory 避免"可变默认值"错误
    paths: Paths = field(default_factory=Paths)
    time_soc: TimeSOC = field(default_factory=TimeSOC)
    charge_queue: ChargingAndQueue = field(default_factory=ChargingAndQueue)
    costs_equity: CostsAndEquity = field(default_factory=CostsAndEquity)
    end_soc: EndSOCConstraints = field(default_factory=EndSOCConstraints)
    pruning: PruningRules = field(default_factory=PruningRules)
    solver: SolverConfig = field(default_factory=SolverConfig)
    flags: ModelFlags = field(default_factory=ModelFlags)
    arc_control: ArcTypeControl = field(default_factory=ArcTypeControl)
    energy: EnergyRates = field(default_factory=EnergyRates)
    basic: BasicConfig = field(default_factory=BasicConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)

    # C_level 的 (t,i,j) 权重覆盖（需要时自行填充）
    unmet_weights_overrides: Optional[Dict[int, Dict[Any, float]]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

# 便捷获取（模块级单例）
_cfg_singleton = NetworkConfig()

def get_network_config() -> NetworkConfig:
    return _cfg_singleton

if __name__ == "__main__":
    import pprint
    pprint.pp(get_network_config().as_dict())
