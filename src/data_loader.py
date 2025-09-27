# data_loader.py
# 基本数据读取函数的集中管理
# 为 reachability、arc_generators、costs 等模块提供统一的数据读取接口
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from _01_network_config import get_network_config as get_config
from grid_utils import GridIndexers, first_present

def load_base_ij(cfg) -> pd.DataFrame:
    """
    读取基础区域间距离矩阵 (base_ij)
    
    Args:
        cfg: 配置对象
        
    Returns:
        包含 i, j, dist_km, base_minutes 列的 DataFrame
    """
    df = pd.read_parquet(cfg.paths.base_ij)
    col_i = first_present(df.columns, ["i"])
    col_j = first_present(df.columns, ["j"])
    col_min = first_present(df.columns, ["base_minutes", "minutes"])
    
    if None in (col_i, col_j):
        raise ValueError("base_ij.parquet 需包含列: i, j。")

    out_cols = [col_i, col_j]
    out_rename = {col_i: "i", col_j: "j"}
    if col_min is not None:
        out_cols.append(col_min)
        out_rename[col_min] = "base_minutes"
    
    out = df[out_cols].rename(columns=out_rename)
    out["dist_km"] = pd.to_numeric(df["dist_km"], errors="coerce").fillna(0.0) if "dist_km" in df.columns else 0.0
    out["i"] = out["i"].astype(int)
    out["j"] = out["j"].astype(int)
    out["base_minutes"] = pd.to_numeric(out["base_minutes"], errors="coerce").fillna(0.0)
    
    return out

def load_base_i2k(cfg) -> pd.DataFrame:
    """
    读取基础区域到充电站距离矩阵 (base_i2k)
    
    Args:
        cfg: 配置对象
        
    Returns:
        包含 i, k, dist_km 列的 DataFrame
    """
    df = pd.read_parquet(cfg.paths.base_i2k)
    col_i = first_present(df.columns, ["i"])
    col_k = first_present(df.columns, ["k"])
    
    if None in (col_i, col_k):
        raise ValueError("base_i2k.parquet 需包含列: i, k。")
    
    out = df[[col_i, col_k]].rename(columns={col_i: "i", col_k: "k"})
    out["dist_km"] = pd.to_numeric(df["dist_km"], errors="coerce").fillna(0.0) if "dist_km" in df.columns else 0.0
    out["i"] = out["i"].astype(int)
    out["k"] = out["k"].astype(int)
    
    return out

def load_stations_mapping(cfg) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    读取充电站到区域和级别的映射
    
    Args:
        cfg: 配置对象
        
    Returns:
        (k_to_zone, k_to_level) 两个映射字典
    """
    st = pd.read_csv(cfg.paths.stations)
    required = ["k", "zone", "level"]
    
    if any(c not in st.columns for c in required):
        raise ValueError("stations.csv 必须包含列: k, zone, level。")
    
    st = st[required].copy()
    st["k"] = st["k"].astype(int)
    st["zone"] = st["zone"].astype(int)
    st["level"] = pd.to_numeric(st["level"], errors="coerce").fillna(3).astype(int)
    
    k_to_zone = dict(zip(st["k"], st["zone"]))
    k_to_level = dict(zip(st["k"], st["level"]))
    
    return k_to_zone, k_to_level

def load_initial_inventory(cfg, gi: GridIndexers) -> pd.DataFrame:
    """
    读取初始车量库存
    
    Args:
        cfg: 配置对象
        gi: 网格索引器
        
    Returns:
        包含 zone, soc, count 列的 DataFrame（仅 t = start_step）
    """
    fp = Path(cfg.paths.fleet_init)
    if not fp.exists():
        raise FileNotFoundError(f"Fleet initial inventory not found: {fp}（请提供 data/fleet_init.csv）")

    df = pd.read_csv(fp)
    col_zone = first_present(df.columns, ["zone"])
    col_soc  = first_present(df.columns, ["soc"])
    col_cnt  = first_present(df.columns, ["count"])

    if col_zone is None or col_soc is None or col_cnt is None:
        raise ValueError("fleet_init.csv 必须包含列: zone(或 taz), soc(或 soc_level), count(或 quantity)。")

    df = df[[col_zone, col_soc, col_cnt]].rename(
        columns={col_zone: "zone", col_soc: "soc", col_cnt: "count"}
    )
    df["zone"]  = df["zone"].astype(int)
    df["soc"]   = df["soc"].astype(int)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0).astype(float)
    df.loc[df["count"] < 0, "count"] = 0.0

    # 聚合重复键
    df = df.groupby(["zone", "soc"], as_index=False)["count"].sum()

    # 校验 zone/soc
    zones_valid = df["zone"].isin(gi.zones)
    socs_valid = df["soc"].isin(gi.socs)
    
    if not zones_valid.all():
        invalid_zones = df[~zones_valid]["zone"].unique()
        print(f"[data_loader] WARN: 发现无效区域 {invalid_zones}，将过滤。")
        df = df[zones_valid]
    
    if not socs_valid.all():
        invalid_socs = df[~socs_valid]["soc"].unique()
        print(f"[data_loader] WARN: 发现无效SOC级别 {invalid_socs}，将过滤。")
        df = df[socs_valid]

    # 补全缺失组合
    all_combinations = pd.MultiIndex.from_product([gi.zones, gi.socs], names=["zone", "soc"])
    df_full = df.set_index(["zone", "soc"]).reindex(all_combinations, fill_value=0.0).reset_index()
    
    return df_full

def load_initial_inventory_parquet(path: str | Path = "data/intermediate/initial_inventory.parquet") -> pd.DataFrame:
    """
    读取初始库存的parquet文件（用于connectivity模块）
    
    Args:
        path: parquet文件路径
        
    Returns:
        包含 zone, soc, t, count 列的 DataFrame
    """
    return pd.read_parquet(str(path), columns=["zone", "soc", "t", "count"])

def validate_data_quality(cfg) -> Dict[str, any]:
    """
    验证数据质量
    
    Args:
        cfg: 配置对象
        
    Returns:
        数据质量报告
    """
    report = {}
    
    # 验证 base_ij
    try:
        ij_df = load_base_ij(cfg)
        report["base_ij"] = {
            "rows": len(ij_df),
            "has_dist_km": "dist_km" in ij_df.columns,
            "has_base_minutes": "base_minutes" in ij_df.columns,
            "dist_km_stats": ij_df["dist_km"].describe().to_dict() if "dist_km" in ij_df.columns else None,
            "base_minutes_stats": ij_df["base_minutes"].describe().to_dict() if "base_minutes" in ij_df.columns else None
        }
    except Exception as e:
        report["base_ij"] = {"error": str(e)}
    
    # 验证 base_i2k
    try:
        i2k_df = load_base_i2k(cfg)
        report["base_i2k"] = {
            "rows": len(i2k_df),
            "has_dist_km": "dist_km" in i2k_df.columns,
            "dist_km_stats": i2k_df["dist_km"].describe().to_dict() if "dist_km" in i2k_df.columns else None,
            "unique_zones": i2k_df["i"].nunique(),
            "unique_stations": i2k_df["k"].nunique()
        }
    except Exception as e:
        report["base_i2k"] = {"error": str(e)}
    
    # 验证 stations
    try:
        k_to_zone, k_to_level = load_stations_mapping(cfg)
        report["stations"] = {
            "total_stations": len(k_to_zone),
            "unique_zones": len(set(k_to_zone.values())),
            "unique_levels": len(set(k_to_level.values())),
            "level_distribution": pd.Series(list(k_to_level.values())).value_counts().to_dict()
        }
    except Exception as e:
        report["stations"] = {"error": str(e)}
    
    return report

def generate_data_summary(cfg) -> pd.DataFrame:
    """
    生成数据摘要报告
    
    Args:
        cfg: 配置对象
        
    Returns:
        数据摘要 DataFrame
    """
    summary_data = []
    
    # base_ij 摘要
    try:
        ij_df = load_base_ij(cfg)
        summary_data.append({
            "dataset": "base_ij",
            "rows": len(ij_df),
            "columns": list(ij_df.columns),
            "status": "OK"
        })
    except Exception as e:
        summary_data.append({
            "dataset": "base_ij",
            "rows": 0,
            "columns": [],
            "status": f"ERROR: {e}"
        })
    
    # base_i2k 摘要
    try:
        i2k_df = load_base_i2k(cfg)
        summary_data.append({
            "dataset": "base_i2k",
            "rows": len(i2k_df),
            "columns": list(i2k_df.columns),
            "status": "OK"
        })
    except Exception as e:
        summary_data.append({
            "dataset": "base_i2k",
            "rows": 0,
            "columns": [],
            "status": f"ERROR: {e}"
        })
    
    # stations 摘要
    try:
        k_to_zone, k_to_level = load_stations_mapping(cfg)
        summary_data.append({
            "dataset": "stations",
            "rows": len(k_to_zone),
            "columns": ["k", "zone", "level"],
            "status": "OK"
        })
    except Exception as e:
        summary_data.append({
            "dataset": "stations",
            "rows": 0,
            "columns": [],
            "status": f"ERROR: {e}"
        })
    
    return pd.DataFrame(summary_data)

# 公共包装函数，遵循用户偏好
def generate_base_ij_data(cfg=None) -> pd.DataFrame:
    """
    公共包装函数：读取 base_ij 数据
    
    Args:
        cfg: 配置对象，默认为 None 时自动获取
        
    Returns:
        base_ij DataFrame
    """
    if cfg is None:
        cfg = get_config()
    return load_base_ij(cfg)

def generate_base_i2k_data(cfg=None) -> pd.DataFrame:
    """
    公共包装函数：读取 base_i2k 数据
    
    Args:
        cfg: 配置对象，默认为 None 时自动获取
        
    Returns:
        base_i2k DataFrame
    """
    if cfg is None:
        cfg = get_config()
    return load_base_i2k(cfg)

def generate_stations_mapping(cfg=None) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    公共包装函数：读取充电站映射
    
    Args:
        cfg: 配置对象，默认为 None 时自动获取
        
    Returns:
        (k_to_zone, k_to_level) 映射字典
    """
    if cfg is None:
        cfg = get_config()
    return load_stations_mapping(cfg)

def generate_initial_inventory(cfg=None, gi=None) -> pd.DataFrame:
    """
    公共包装函数：读取初始库存
    
    Args:
        cfg: 配置对象，默认为 None 时自动获取
        gi: 网格索引器，默认为 None 时自动加载
        
    Returns:
        初始库存 DataFrame
    """
    if cfg is None:
        cfg = get_config()
    if gi is None:
        from grid_utils import load_indexer
        gi = load_indexer()
    return load_initial_inventory(cfg, gi)

def generate_data_quality_report(cfg=None) -> Dict[str, any]:
    """
    公共包装函数：生成数据质量报告
    
    Args:
        cfg: 配置对象，默认为 None 时自动获取
        
    Returns:
        数据质量报告
    """
    if cfg is None:
        cfg = get_config()
    return validate_data_quality(cfg)

def load_od_matrix(cfg=None) -> pd.DataFrame:
    """
    加载OD矩阵数据
    
    Args:
        cfg: 配置对象，如果为None则使用默认配置
        
    Returns:
        包含 t, i, j, demand 列的 DataFrame
    """
    if cfg is None:
        cfg = get_config()
    
    df = pd.read_parquet(cfg.paths.od_matrix)
    
    # 确保必要的列存在
    required_cols = ['t', 'i', 'j', 'demand']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"OD矩阵缺少列: {missing_cols}")
    
    # 数据类型转换
    df['t'] = df['t'].astype(int)
    df['i'] = df['i'].astype(int)
    df['j'] = df['j'].astype(int)
    df['demand'] = pd.to_numeric(df['demand'], errors='coerce').fillna(0.0)
    
    return df

def load_stations_dataframe(cfg=None) -> pd.DataFrame:
    """
    加载完整的充电站DataFrame
    
    Args:
        cfg: 配置对象，如果为None则使用默认配置
        
    Returns:
        包含所有充电站信息的 DataFrame
    """
    if cfg is None:
        cfg = get_config()
    
    df = pd.read_csv(cfg.paths.stations)
    
    # 确保必要的列存在
    required_cols = ['k', 'zone', 'level', 'plugs', 'util_factor']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"充电站文件缺少列: {missing_cols}")
    
    # 数据类型转换
    df['k'] = df['k'].astype(int)
    df['zone'] = df['zone'].astype(int)
    
    # 处理level列：将'l2'/'l3'等字符串转换为数字
    def parse_level(level_val):
        if pd.isna(level_val):
            return 3  # 默认L3
        level_str = str(level_val).lower()
        if level_str.startswith('l'):
            try:
                return int(level_str[1:])
            except ValueError:
                return 3  # 默认L3
        else:
            try:
                return int(level_val)
            except ValueError:
                return 3  # 默认L3
    
    df['level'] = df['level'].apply(parse_level).astype(int)
    df['plugs'] = df['plugs'].astype(int)
    df['util_factor'] = pd.to_numeric(df['util_factor'], errors='coerce').fillna(1.0)
    
    return df

def load_fleet_init(cfg=None) -> pd.DataFrame:
    """
    加载初始车队数据
    
    Args:
        cfg: 配置对象，如果为None则使用默认配置
        
    Returns:
        包含 zone, soc, count 列的 DataFrame
    """
    if cfg is None:
        cfg = get_config()
    
    fleet_path = Path(cfg.paths.fleet_init)
    if not fleet_path.exists():
        # 如果没有初始车队文件，返回空DataFrame
        return pd.DataFrame(columns=['zone', 'soc', 'count'])
    
    df = pd.read_csv(fleet_path)
    
    # 确保必要的列存在
    required_cols = ['zone', 'soc', 'count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"初始车队文件缺少列: {missing_cols}")
    
    # 数据类型转换
    df['zone'] = df['zone'].astype(int)
    df['soc'] = df['soc'].astype(int)
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0.0)
    
    return df

def load_or_build_charging_profile(cfg) -> pd.DataFrame:
    """
    优先读取 data/charging_profile.csv：
      列: level, from_soc, to_soc, tau_chg_minutes
    若不存在，则构建线性近似（L3: 60min/100%，L2: 300min/100%）
    
    Args:
        cfg: 配置对象
        
    Returns:
        充电曲线 DataFrame
    """
    prof_path = Path("data/charging_profile.csv")
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        needed = ["level", "from_soc", "to_soc", "tau_chg_minutes"]
        if any(c not in prof.columns for c in needed):
            raise ValueError(f"{prof_path} 必须包含列: {needed}")
        prof["from_soc"] = prof["from_soc"].astype(int)
        prof["to_soc"] = prof["to_soc"].astype(int)
        prof["level"] = prof["level"].astype(int)
        prof["tau_chg_minutes"] = prof["tau_chg_minutes"].astype(float)
        return prof

    warnings.warn(
        "[data_loader] charging_profile.csv 未找到；使用线性近似 (L3:60min/100%, L2:300min/100%).",
        RuntimeWarning,
    )
    rows = []
    linear_rates = {3: 0.6, 2: 3.0}  # min per 1% soc
    for level, m_per_pct in linear_rates.items():
        for l_from in range(0, 100, 10):
            for l_to in range(l_from + 10, 101, 10):
                rows.append({
                    "level": level,
                    "from_soc": l_from,
                    "to_soc": l_to,
                    "tau_chg_minutes": m_per_pct * (l_to - l_from),
                })
    return pd.DataFrame(rows)

def load_station_capacity_map(cfg):
    import numpy as np, pandas as pd
    df = pd.read_csv(cfg.paths.stations)
    # 兼容列名
    c_k   = next(c for c in df.columns if c.lower() in {"k","station","station_id"})
    c_p   = next(c for c in df.columns if c.lower() in {"plugs","plug","stalls"})
    c_u   = "util_factor" if "util_factor" in df.columns else None

    plugs = pd.to_numeric(df[c_p], errors="coerce").fillna(0).clip(lower=0)
    util  = pd.to_numeric(df[c_u], errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0) if c_u else 1.0

    relax = float(cfg.charge_queue.queue_relax_factor or 1.0)
    # 有两种口径可选：不超物理（floor）或允许放松（ceil）
    eff = np.maximum(1, np.floor(plugs * util * relax)).astype(int)   # 保守：不超过物理容量
    # 若希望“放松”更明显，可改成：eff = np.maximum(1, np.ceil(plugs * util * relax)).astype(int)

    cap_map = dict(zip(df[c_k].astype(int), eff.astype(int)))
    return cap_map
