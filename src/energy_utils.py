# energy_utils.py
# 能耗计算共用底层逻辑函数
# 为 service、reposition、charging 三种弧类型提供统一的能耗计算接口
from __future__ import annotations
import math
import os
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from _01_network_config import get_network_config as get_config

# 全局警告记录，避免重复警告
_warned = set()

def get_energy_rate(cfg, name: str, t: int, default: float = 0.0) -> float:
    """
    按时间步 t 获取能耗参数，支持多种数据源：
    1. 优先读取 cfg.paths.coeff_energy 指定的 CSV 文件
    2. 回退到 cfg.energy.* 配置参数
    3. 最后使用默认值
    
    Args:
        cfg: 配置对象
        name: 能耗参数名称 (如 "de_per_km_srv", "de_per_km_rep", "de_per_km_tochg")
        t: 时间步
        default: 默认值
        
    Returns:
        能耗率 (每公里消耗的 SOC 百分比)
    """
    coeff_path = cfg.paths.coeff_energy
    if os.path.exists(coeff_path):
        try:
            df = pd.read_csv(coeff_path)
            if "t" in df.columns:
                row = df[df["t"] == t]
                if not row.empty and name in row.columns:
                    return float(row[name].iloc[0])
                else:
                    if f"coeff_energy_missing_{name}_t{t}" not in _warned:
                        warnings.warn(f"[energy_utils] coeff_energy.csv 缺 t={t} 或列 {name}，回退默认。", RuntimeWarning)
                        _warned.add(f"coeff_energy_missing_{name}_t{t}")
                    return default
            else:
                if "coeff_energy_no_t_col" not in _warned:
                    warnings.warn("[energy_utils] coeff_energy.csv 缺少 t 列，回退默认。", RuntimeWarning)
                    _warned.add("coeff_energy_no_t_col")
                return default
        except Exception as e:
            if f"coeff_energy_read_error_{name}" not in _warned:
                warnings.warn(f"[energy_utils] coeff_energy.csv 读取失败: {e}，回退默认。", RuntimeWarning)
                _warned.add(f"coeff_energy_read_error_{name}")
            return default
    
    # 回退到配置参数
    ns = cfg.energy if hasattr(cfg, "energy") else None
    if ns is None:
        if "no_energy_ns" not in _warned:
            warnings.warn("[energy_utils] 未在 config 中找到 energy.* 参数；能耗将默认为 0。", RuntimeWarning)
            _warned.add("no_energy_ns")
        return default
    
    try:
        return float(ns.__dict__.get(name, default))
    except Exception:
        return default

def compute_energy_consumption(dist_km: float, per_km: float, soc_step: int) -> int:
    """
    计算单步能耗消耗（SOC 点数）
    
    Args:
        dist_km: 距离（公里）
        per_km: 每公里能耗率
        soc_step: SOC 步长
        
    Returns:
        能耗消耗（SOC 点数），上取整到 soc_step 的倍数
    """
    raw_pct = max(0.0, dist_km * per_km)
    if raw_pct <= 0:
        return 0
    de = int(math.ceil(raw_pct / soc_step) * soc_step)
    return int(min(de, 100))

def compute_multi_timestep_energy_consumption(
    cfg, 
    dist_km: float, 
    start_t: int, 
    tau_steps: int, 
    soc_step: int, 
    energy_type: str
) -> int:
    """
    计算多时间步的能耗消耗（每步耗能率可随时间变化）
    
    Args:
        cfg: 配置对象
        dist_km: 距离（公里）
        start_t: 开始时间步
        tau_steps: 时间步数
        soc_step: SOC 步长
        energy_type: 能耗类型 ("de_per_km_srv", "de_per_km_rep", "de_per_km_tochg")
        
    Returns:
        总能耗消耗（SOC 点数）
    """
    if tau_steps <= 0 or dist_km <= 0:
        return 0
    
    if tau_steps == 1:
        de_km = get_energy_rate(cfg, energy_type, start_t, 0.0)
        return compute_energy_consumption(dist_km, de_km, soc_step)
    
    # 多步情况：将距离平均分配到各时间步
    dist_per_step = dist_km / tau_steps
    total_de = 0.0
    
    for i in range(tau_steps):
        t = start_t + i
        de_km = get_energy_rate(cfg, energy_type, t, 0.0)
        total_de += dist_per_step * de_km
    
    de = int(math.ceil(total_de / soc_step) * soc_step)
    return min(de, 100)

def estimate_travel_time_steps(base_minutes: float, dt_minutes: int) -> int:
    """
    根据基础行驶时间估算时间步数
    
    Args:
        base_minutes: 基础行驶时间（分钟）
        dt_minutes: 时间步长（分钟）
        
    Returns:
        时间步数
    """
    return int(math.ceil(base_minutes / dt_minutes))

def compute_travel_time_from_distance(dist_km: float, avg_speed_kmh: float) -> float:
    """
    根据距离和平均速度计算行驶时间
    
    Args:
        dist_km: 距离（公里）
        avg_speed_kmh: 平均速度（公里/小时）
        
    Returns:
        行驶时间（分钟）
    """
    if dist_km <= 0 or avg_speed_kmh <= 0:
        return 0.0
    return (dist_km / avg_speed_kmh) * 60.0

def get_soc_step_from_levels(soc_levels: List[int]) -> int:
    """
    从 SOC 级别列表中计算步长
    
    Args:
        soc_levels: SOC 级别列表
        
    Returns:
        SOC 步长
    """
    if len(soc_levels) <= 1:
        return 100
    
    soc_array = np.array(soc_levels, dtype=int)
    return int(np.diff(soc_array).min())

def validate_energy_parameters(cfg) -> Dict[str, bool]:
    """
    验证能耗参数配置的完整性
    
    Args:
        cfg: 配置对象
        
    Returns:
        验证结果字典
    """
    results = {}
    
    # 检查 coeff_energy.csv 文件
    coeff_path = cfg.paths.coeff_energy
    results["has_coeff_energy_file"] = os.path.exists(coeff_path)
    
    if results["has_coeff_energy_file"]:
        try:
            df = pd.read_csv(coeff_path)
            results["has_t_column"] = "t" in df.columns
            results["has_energy_columns"] = any(col.startswith("de_per_km") for col in df.columns)
        except Exception:
            results["coeff_energy_readable"] = False
    else:
        results["has_t_column"] = False
        results["has_energy_columns"] = False
        results["coeff_energy_readable"] = False
    
    # 检查配置参数
    energy_ns = cfg.energy if hasattr(cfg, "energy") else None
    results["has_energy_config"] = energy_ns is not None
    
    return results

def generate_energy_consumption_report(cfg, sample_times: List[int] = None) -> pd.DataFrame:
    """
    生成能耗参数报告，用于调试和验证
    
    Args:
        cfg: 配置对象
        sample_times: 采样时间步列表，默认为 [0, 12, 24]
        
    Returns:
        能耗参数报告 DataFrame
    """
    if sample_times is None:
        sample_times = [0, 12, 24]
    
    energy_types = ["de_per_km_srv", "de_per_km_rep", "de_per_km_tochg"]
    rows = []
    
    for t in sample_times:
        for energy_type in energy_types:
            rate = get_energy_rate(cfg, energy_type, t, 0.0)
            rows.append({
                "t": t,
                "energy_type": energy_type,
                "rate": rate,
                "source": "coeff_energy.csv" if os.path.exists(cfg.paths.coeff_energy) else "config.energy"
            })
    
    return pd.DataFrame(rows)

# 公共包装函数，遵循用户偏好
def generate_energy_consumption(dist_km: float, start_t: int, tau_steps: int, 
                              energy_type: str, cfg=None) -> int:
    """
    公共包装函数：计算能耗消耗
    
    Args:
        dist_km: 距离（公里）
        start_t: 开始时间步
        tau_steps: 时间步数
        energy_type: 能耗类型
        cfg: 配置对象，默认为 None 时自动获取
        
    Returns:
        能耗消耗（SOC 点数）
    """
    if cfg is None:
        cfg = get_config()
    
    soc_levels = cfg.soc_levels if hasattr(cfg, "soc_levels") else list(range(0, 101, 10))
    soc_step = get_soc_step_from_levels(soc_levels)
    
    return compute_multi_timestep_energy_consumption(
        cfg, dist_km, start_t, tau_steps, soc_step, energy_type
    )

def generate_travel_time_estimation(dist_km: float, base_minutes: float = None, 
                                   avg_speed_kmh: float = None, cfg=None) -> Tuple[float, int]:
    """
    公共包装函数：估算行驶时间
    
    Args:
        dist_km: 距离（公里）
        base_minutes: 基础行驶时间（分钟），优先级最高
        avg_speed_kmh: 平均速度（公里/小时）
        cfg: 配置对象
        
    Returns:
        (行驶时间分钟, 时间步数)
    """
    if cfg is None:
        cfg = get_config()
    
    if base_minutes is not None:
        travel_minutes = base_minutes
    elif avg_speed_kmh is not None:
        travel_minutes = compute_travel_time_from_distance(dist_km, avg_speed_kmh)
    else:
        # 从配置获取默认平均速度
        avg_speed = float(cfg.basic.avg_speed_kmh) if hasattr(cfg, "basic") and hasattr(cfg.basic, "avg_speed_kmh") else 30.0
        travel_minutes = compute_travel_time_from_distance(dist_km, avg_speed)
    
    dt_minutes = int(cfg.time_soc.dt_minutes)
    tau_steps = estimate_travel_time_steps(travel_minutes, dt_minutes)
    
    return travel_minutes, tau_steps
