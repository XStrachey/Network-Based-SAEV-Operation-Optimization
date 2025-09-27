# _01_network/_01_network_config.py
# 网络求解方案的独立配置

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

# -------------------------
# 路径（仅必需外部输入 + 输出）
# -------------------------
@dataclass
class Paths:
    zones: str = "../data/zones.csv"                 # 列: zone, lat, lon
    stations: str = "../data/stations.csv"           # 列: k, zone, level, lat, lon, plugs, util_factor
    od_matrix: str = "../data/od_matrix.parquet"     # 列: t, i, j, demand
    base_ij: str = "../data/base_ij.parquet"         # 列: i, j, base_minutes[, dist_km]
    base_i2k: str = "../data/base_i2k.parquet"       # 列: i, k, base_minutes[, dist_km]
    coeff_schedule: str = "../data/coeff_schedule.csv"  # 列: t, gamma_rep_p, beta_chg_p1, beta_chg_p2
    coeff_energy: str = "../data/coeff_energy.csv"      # 列: t, de_per_km_srv, de_per_km_rep, de_per_km_tochg
    fleet_init: str = "../data/fleet_init.csv"

# -------------------------
# 时间与 SOC 离散
# -------------------------
@dataclass
class TimeSOC:
    dt_minutes: int = 15
    start_step: int = 1
    end_step: int = 96
    window_length: int = 24           # 窗口长度 H
    overhang_steps: int = 2          # Halo：窗外可见/可到达的"缓冲步数"（建议 2~6）
    roll_step: int = 1
    soc_levels: List[int] = None

    def __post_init__(self):
        if self.soc_levels is None:
            # 0, 5, 10, ..., 100
            self.soc_levels = list(range(0, 101, 5))

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

    # —— 服务奖励 / 未满足惩罚 —— 
    unmet_weight_default: float = 1.0    # 06 的 svc_gate 奖励可等价为 -VOT*unmet_weight_default
    alpha_unmet: float = 1.0              # 为与单期保持一致，单期里常写作 VOT*alpha_unmet

    # —— 新目标函数：收益项系数（请保持非负）——
    gamma_reposition_reward: float = 1.0  # 重定位收益系数 γ_rep（施加在 reposition 弧，按目的地 j 与 t）
    beta_chg_reward: float = 0.1                   # 充电收益系数 α_chg（施加在 chg_occ/step 弧，按 ΔSOC）

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
    max_reposition_tt: float = 30.0      # 若 τ_rep > 30 min，剪枝
    min_soc_for_reposition: int = 20     # SOC% 低于阈值不允许重定位
    reposition_nearest_zone_n: int = 8
    charge_nearest_station_n: int = 8
    max_service_radius_zones: int = 0    # 0=不限制；>0 时仅保留邻近服务弧（在 04 中用）

# -------------------------
# 求解器
# -------------------------
@dataclass
class SolverConfig:
    solver_name: str = "glpk"
    time_limit_sec: int = 1800           # 默认延长到 30min，更符合你现在的规模
    mip_gap: float = 0.01
    threads: int = 0
    verbose: bool = True
    RELAX_BINARIES: bool = True          # 仅控制是否放松二元为 [0,1]
    use_nopresol: bool = True            # 若上层支持，传递给 GLPK 以便超时仍能导出可行解
    check_negative_cycles: bool = True   # 是否在求解前检测负环

# -------------------------
# 模型结构开关
# -------------------------
@dataclass
class ModelFlags:
    # —— 新增：06 成本组件开关（便于 A/B 测试）——
    enable_service_reward: bool = True       # svc_gate 奖励（等价未满足惩罚）
    enable_reposition_reward: bool = True    # γ_rep * zone_value 的重定位收益
    enable_charging_reward: bool = True      # α_chg * ΔSOC 的充电收益

# -------------------------
# 动态弧生成的配置（Halo 等）
# -------------------------
@dataclass
class ArcGenConfig:
    mode: str = "dynamic"                # 'dynamic' | 'static'
    # None 表示沿用 time_soc.overhang_steps；否则覆盖
    halo_steps: Optional[int] = None

@dataclass
class EnergyRates:
    de_per_km_srv: float = 0.2
    de_per_km_rep: float = 0.2
    de_per_km_tochg: float = 0.2

@dataclass
class BasicConfig:
    avg_speed_kmh: float = 30.0

# -------------------------
# R1重定位弧定向裁剪配置
# -------------------------
@dataclass
class R1PruneConfig:
    enabled: bool = True                    # 总开关
    delta: float = 0.0                     # 判定阈值：Δ=DP(j,t_to)-DP(i,t_from) ≥ delta 才定向
    epsilon: float = 0.05                  # 不确定带：|Δ| ≤ epsilon 则保留双向
    keep_reverse_ratio: float = 0.10       # 兜底：按成本/距离对每对(i,j)保留少量反向边
    min_outdeg: int = 2                    # 每区/时段最小出度下限（重定位）
    min_indeg: int = 2                     # 每区/时段最小入度下限（重定位）
    supply_mode: str = "none"              # 供给估计：none | prev_solution（可后续扩展）
    random_seed: int = 13                  # 随机兜底的可复现性

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
    energy: EnergyRates = field(default_factory=EnergyRates)
    basic: BasicConfig = field(default_factory=BasicConfig)
    r1_prune: R1PruneConfig = field(default_factory=R1PruneConfig)

    arcgen: ArcGenConfig = field(default_factory=ArcGenConfig)

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
