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
from config.network_config import get_network_config as get_config

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