# _04_arc_generators.py  (modified for pure-network formulation with self-loop removal)
# 生成弧：idle / service(拆为: svc_enter, svc_gate, svc_exit) /
#        reposition / charging(拆为: tochg, chg_enter, chg_occ, chg_step)
# 支持两种模式：
#   (A) 批量静态预生成（main）：落盘到 data/intermediate/*.parquet
#   (B) 窗口动态生成（generate_arcs_for_window）：仅生成“出发在窗内”的弧，允许到达跨窗（Halo）
#
# 关键变化（纯网络化）：
#   1) 服务需求采用 "闸门弧"：对每个 (i,j,t) 引入 svc_gate 单条容量弧（cap = demand，成本稍后由 _06_costs 设为 -reward），
#      并用 svc_enter / svc_exit 把原服务路径拆成三段，去除了“组约束”。
#   2) 充电并发采用 "占用链"：为每个 (k,p) 建立 chg_occ 单条容量弧（cap = 插枪数），
#      车辆每充电 1 步都必须穿过该弧；SOC 提升在最后一步一次性跳升（也可改为逐步上升）。
#   3) **新增：建弧阶段统一移除自环**（from_node_id == to_node_id）以避免负成本自环产生“空刷奖励”。
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import warnings
import pickle
import json
import os
import hashlib
from dataclasses import dataclass

# 兼容两种配置模块路径
try:
    from network_config import get_network_config as get_config  # 优先
except Exception:  # noqa
    try:
        from network.network_config import get_network_config as get_config  # 备选
    except Exception:  # noqa
        from _01_network_config import get_network_config as get_config  # 老路径

from energy_utils import (
    get_energy_rate,
    compute_energy_consumption,
    compute_multi_timestep_energy_consumption,
)
from grid_utils import GridIndexers, load_indexer, ensure_dir, first_present
from data_loader import (
    load_base_ij,
    load_base_i2k,
    load_stations_mapping,
    load_or_build_charging_profile,
    load_station_capacity_map,
)

# ----------------------------
# Reachability 预裁剪加载
# ----------------------------
def load_reachability_with_time(path="data/intermediate/reachability.parquet") -> set:
    """返回可达(zone, soc, time)的集合，用于更精确的过滤"""
    df = pd.read_parquet(path)
    return set(
        (int(r["zone"]), int(r["soc"]), int(r["t"]))
        for _, r in df.iterrows()
        if r["reachable"] == 1
    )

# ----------------------------
# 工具/加载
# ----------------------------
def _make_arc_id(df: pd.DataFrame, key_cols) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="int64")
    key = df[key_cols].astype(str).agg("|".join, axis=1)
    h = pd.util.hash_pandas_object(key, index=False).astype("int64")
    return (h & np.int64(0x7FFFFFFFFFFFFFFF)).astype("int64")  # 保证非负

def _attach_arc_id(df: pd.DataFrame, key_cols, drop_dups=True) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["arc_id"] = _make_arc_id(out, key_cols)
    if drop_dups:
        out = out.drop_duplicates(subset=["arc_id"], keep="first")
    return out

def _pseudo_node_id(kind: str, *keys) -> int:
    """
    生成稳定的伪节点 id（负数），避免与网格节点冲突。
    kind: 如 'svc_in','svc_out','q_in','q_out'
    keys: 唯一键，如 (i,j,t) 或 (k,p)
    """
    s = f"{kind}|" + "|".join(map(str, keys))
    # 使用 blake2b 生成 8 字节摘要，转 int，再取负
    digest = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
    val = int.from_bytes(digest, byteorder='big', signed=False) & 0x7FFFFFFFFFFFFFFF
    return -int(val)

def load_nodes_df() -> pd.DataFrame:
    return pd.read_parquet("data/intermediate/nodes.parquet")

def _drop_self_loops(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """删除 from_node_id == to_node_id 的自环；若 verbose 开启则打印统计。"""
    if (
        df is None
        or df.empty
        or "from_node_id" not in df.columns
        or "to_node_id" not in df.columns
    ):
        return df
    m = (df["from_node_id"] == df["to_node_id"])
    n = int(m.sum())
    if n > 0 and get_config().solver.verbose:
        print(f"[arcgen] drop {n} self-loops in {name}")
    return df.loc[~m].copy()

# ----------------------------
# 充电/站点辅助
# ----------------------------
def _load_zone_station_data(cfg) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    读取 _03_reachability.py 产物：
      - zone_station_best.parquet: 列含 i,k,dist_km,tau_steps
      - nearest_stations.json: {i: [k1,k2,...]}
    """
    out_dir = Path("data/intermediate")
    best_path = out_dir / "zone_station_best.parquet"
    nearest_path = out_dir / "nearest_stations.json"

    if not best_path.exists():
        raise FileNotFoundError(f"zone_station_best.parquet not found at {best_path}. 请先运行 _03_reachability.py")
    if not nearest_path.exists():
        raise FileNotFoundError(f"nearest_stations.json not found at {nearest_path}. 请先运行 _03_reachability.py")

    best_df = pd.read_parquet(best_path)
    with open(nearest_path, 'r') as f:
        nearest_data = json.load(f)
    nearest_map = {int(k): v for k, v in nearest_data.items()}

    return best_df, nearest_map

# ============================================================
# ================== 1) 基础四类弧（按全局） ==================
# ============================================================
def build_idle_arcs(cfg, gi: GridIndexers, reachable_set: set,
                    t0: int = None, t_hi: int = None, B: int = None) -> pd.DataFrame:
    """
    idle 弧： (i,t,l) -> (i,t+1,l)
    """
    nodes = pd.read_parquet("data/intermediate/nodes.parquet", columns=["node_id", "zone", "t", "soc"])

    if t0 is None:  # 全生成模式
        max_t = gi.times[-1]
        nodes_next = nodes[nodes["t"] + 1 <= max_t].copy()
    else:  # 窗口模式
        max_dep_t = min(t_hi - 1, gi.times[-1] - 1)
        nodes_next = nodes[(nodes["t"] >= t0) & (nodes["t"] <= max_dep_t)].copy()

    if nodes_next.empty:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","i","t","l","tau"])

    nodes_next["to_node_id"] = nodes_next.apply(
        lambda r: gi.id_of(int(r["zone"]), int(r["t"] + 1), int(r["soc"])), axis=1
    )
    idle = nodes_next.rename(columns={"node_id": "from_node_id", "zone": "i", "t": "t", "soc": "l"})
    idle["arc_type"] = "idle"
    idle["tau"] = 1
    out = idle[["arc_type", "from_node_id", "to_node_id", "i", "t", "l", "tau"]]
    return _drop_self_loops(out, "idle")

def build_service_arcs(cfg, gi: GridIndexers, reachable_set: set,
                       t0: int = None, t_hi: int = None, B: int = None) -> pd.DataFrame:
    """
    纯网络化的 service 弧：
      - svc_enter: (i,t,l) -> (svc_in[i,j,t])，tau=0
      - svc_gate:  (svc_in[i,j,t]) -> (svc_out[i,j,t])，唯一容量弧（cap_hint = demand，成本稍后设为 -reward）
      - svc_exit:  (svc_out[i,j,t]) -> (j,t+tau,l-de)（带时间推进与 SOC 下降）
    """
    from data_loader import load_od_matrix
    od = load_od_matrix(cfg)
    od = od[od["demand"] > 0].copy()

    # 时间过滤
    if t0 is not None:  # 窗口模式
        od = od[(od["t"] >= t0) & (od["t"] <= t_hi - 1)]

    if od.empty:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","req_key","cap_hint"])

    bij = load_base_ij(cfg)
    srv = od.merge(bij, on=["i", "j"], how="inner")

    dt = int(cfg.time_soc.dt_minutes)
    arrival_end = min(t_hi + int(B) if (t_hi is not None and B is not None) else gi.times[-1], gi.times[-1])
    soc_levels = np.array(gi.socs, dtype=int)
    soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100

    rows = []
    for _, r in srv.iterrows():
        i, j, t = int(r["i"]), int(r["j"]), int(r["t"])
        D_ijt = float(r["demand"])
        dist_km  = float(r.get("dist_km", 0.0))
        # 基于距离和速度计算耗时，而不是直接使用base_minutes
        avg_speed_kmh = float(cfg.basic.avg_speed_kmh)
        travel_time_minutes = (dist_km / avg_speed_kmh) * 60.0 if avg_speed_kmh > 0 else 0.0
        tau = int(math.ceil(travel_time_minutes / dt))
        t2 = t + tau
        if t2 > arrival_end or tau <= 0:
            continue
        de = compute_multi_timestep_energy_consumption(cfg, dist_km, t, tau, soc_step, "de_per_km_srv")
        feas_soc = soc_levels[soc_levels >= de]
        if len(feas_soc) == 0:
            continue

        # 构造唯一的闸门节点
        in_id  = _pseudo_node_id("svc_in",  i, j, t)
        out_id = _pseudo_node_id("svc_out", i, j, t)
        req_key = f"{i}-{j}-{t}"

        # gate 弧（单条容量弧，tau=0; cap_hint = D_ijt）
        rows.append(("svc_gate", in_id, out_id, i, j, t, None, 0, 0, req_key, D_ijt))

        # enter/exit 弧（按可行 SOC 生成多条）
        for l in feas_soc:
            if (i, l, t) not in reachable_set or (j, l - de, t2) not in reachable_set:
                continue
            from_id = gi.id_of(i, t, l)
            to_id = gi.id_of(j, t2, l - de)
            # enter： (grid) -> (svc_in)
            rows.append(("svc_enter", from_id, in_id, i, j, t, int(l), 0, 0, req_key, None))
            # exit：  (svc_out) -> (grid arrival)
            rows.append(("svc_exit",  out_id, to_id, i, j, t, int(l), tau, int(de), req_key, None))

    if not rows:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","req_key","cap_hint"])

    df = pd.DataFrame(rows, columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","req_key","cap_hint"])
    # 额外断言：gate 不应为自环（svc_in 与 svc_out 定义不同）
    if not df.empty:
        gate = df["arc_type"].eq("svc_gate")
        if gate.any():
            assert (df.loc[gate, "from_node_id"] != df.loc[gate, "to_node_id"]).all(), "svc_gate should never be a self-loop"
    return _drop_self_loops(df, "service")

def build_reposition_arcs(cfg, gi: GridIndexers, reachable_set: set,
                          t0: int = None, t_hi: int = None, B: int = None) -> pd.DataFrame:
    """
    reposition 弧：需求驱动的生成策略
    
    只对有重定位需求的OD对生成弧，显著减少弧数量
    """
    return _build_reposition_arcs_demand_driven(cfg, gi, reachable_set, t0, t_hi, B)


def _compute_reposition_demand(cfg, t0: int = None, t_hi: int = None) -> pd.DataFrame:
    """
    计算重定位需求矩阵
    
    基于服务需求的时空分布和供需不平衡来预测重定位需求
    """
    # 加载服务需求数据作为基础
    from data_loader import load_od_matrix
    od = load_od_matrix(cfg)
    
    # 时间过滤
    if t0 is not None and t_hi is not None:
        od = od[(od["t"] >= t0) & (od["t"] <= t_hi - 1)]
    
    if od.empty:
        return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
    
    reposition_demand = []
    
    # 方法1：基于高服务需求的OD对生成重定位需求
    high_demand_threshold = od['demand'].quantile(0.8)  # 前20%的高需求
    high_demand_pairs = od[od['demand'] > high_demand_threshold].copy()
    
    reposition_ratio = getattr(cfg.pruning, 'reposition_demand_ratio', 0.3)
    
    for _, row in high_demand_pairs.iterrows():
        i, j, t = int(row['i']), int(row['j']), int(row['t'])
        
        # 重定位需求 = 服务需求的一个比例
        reposition_demand_val = row['demand'] * reposition_ratio
        
        if reposition_demand_val > 0:
            reposition_demand.append({
                'i': i,
                'j': j, 
                't': t,
                'reposition_demand': reposition_demand_val
            })
    
    # 方法2：基于供需不平衡的逆向重定位
    # 计算每个区域在每个时刻的需求和供给不平衡
    zone_demand = od.groupby(['i', 't'])['demand'].sum().reset_index()
    zone_demand.columns = ['zone', 't', 'total_demand']
    
    # 简化假设：供给相对均匀分布
    total_supply = 200.0  # 从配置或数据中获取
    num_zones = od['i'].nunique()
    base_supply_per_zone = total_supply / num_zones
    
    # 直接添加供给列到需求数据中
    zone_demand['total_supply'] = base_supply_per_zone
    
    # 计算供需不平衡
    zone_balance = zone_demand.copy()
    zone_balance['imbalance'] = zone_balance['total_demand'] - zone_balance['total_supply']
    
    imbalance_threshold = getattr(cfg.pruning, 'reposition_imbalance_threshold', 1.0)
    
    # 从供给过剩的区域到需求过剩的区域
    for _, row in zone_balance.iterrows():
        zone, t = int(row['zone']), int(row['t'])
        imbalance = row['imbalance']
        
        if imbalance < -imbalance_threshold:  # 供给过剩
            # 找到需求过剩的区域作为目标
            demand_surplus = zone_balance[
                (zone_balance['t'] == t) & 
                (zone_balance['imbalance'] > imbalance_threshold)
            ].copy()
            
            if not demand_surplus.empty:
                # 选择不平衡最严重的几个目标区域
                top_targets = demand_surplus.nlargest(3, 'imbalance')
                
                for _, target in top_targets.iterrows():
                    target_zone = int(target['zone'])
                    reposition_demand_val = min(abs(imbalance), target['imbalance']) * 0.5
                    
                    reposition_demand.append({
                        'i': zone,
                        'j': target_zone,
                        't': t,
                        'reposition_demand': reposition_demand_val
                    })
    
    if not reposition_demand:
        return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
    
    demand_df = pd.DataFrame(reposition_demand)
    
    # 去重和聚合
    demand_df = demand_df.groupby(['i', 'j', 't']).agg({
        'reposition_demand': 'sum'
    }).reset_index()
    
    # 过滤掉重定位需求过小的OD对
    min_demand_threshold = getattr(cfg.pruning, 'min_reposition_demand', 0.1)
    demand_df = demand_df[demand_df['reposition_demand'] >= min_demand_threshold]
    
    return demand_df


def _build_reposition_arcs_demand_driven(cfg, gi: GridIndexers, reachable_set: set,
                                       t0: int = None, t_hi: int = None, B: int = None) -> pd.DataFrame:
    """
    需求驱动的重定位弧生成
    """
    # 计算重定位需求
    reposition_demand = _compute_reposition_demand(cfg, t0, t_hi)
    
    if reposition_demand.empty:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","dist_km"])
    
    # 加载距离和时间信息
    bij = load_base_ij(cfg)
    
    # 合并重定位需求与距离信息
    demand_with_dist = reposition_demand.merge(bij, on=['i', 'j'], how='inner')
    
    # 注意：距离过滤由 max_reposition_tt 通过时间约束统一控制，无需额外距离过滤
    
    if demand_with_dist.empty:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","dist_km"])
    
    # 生成弧的逻辑（与传统方法相同）
    return _generate_reposition_arcs_from_pairs(cfg, gi, reachable_set, demand_with_dist, t0, t_hi, B)


# 传统重定位弧生成方法已移除 - 全面采用需求驱动方法


def _generate_reposition_arcs_from_pairs(cfg, gi: GridIndexers, reachable_set: set,
                                        od_pairs_df: pd.DataFrame, t0: int = None, t_hi: int = None, B: int = None) -> pd.DataFrame:
    """
    从OD对生成重定位弧的通用函数
    """
    dt = int(cfg.time_soc.dt_minutes)
    arrival_end = min(t_hi + int(B) if (t_hi is not None and B is not None) else gi.times[-1], gi.times[-1])
    soc_levels = np.array(gi.socs, dtype=int)
    soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100

    max_rep_min = float(cfg.pruning.max_reposition_tt)
    max_rep_steps = int(math.ceil(max_rep_min / dt)) if max_rep_min > 0 else None

    rows = []
    
    for _, r in od_pairs_df.iterrows():
        i, j, t = int(r["i"]), int(r["j"]), int(r["t"])
        dist_km = float(r.get("dist_km", 0.0))
        
        # 基于距离和速度计算耗时
        avg_speed_kmh = float(cfg.basic.avg_speed_kmh)
        travel_time_minutes = (dist_km / avg_speed_kmh) * 60.0 if avg_speed_kmh > 0 else 0.0
        tau = int(math.ceil(travel_time_minutes / dt))
        
        if tau <= 0:
            continue
        if max_rep_steps is not None and tau > max_rep_steps:
            continue
            
        t2 = t + tau
        if t2 > arrival_end:
            continue
            
        de = compute_multi_timestep_energy_consumption(cfg, dist_km, t, tau, soc_step, "de_per_km_rep")
        lmin = max(int(cfg.pruning.min_soc_for_reposition), de)
        feas_soc = soc_levels[soc_levels >= lmin]
        
        for l in feas_soc:
            if (i, l, t) not in reachable_set or (j, l - de, t2) not in reachable_set:
                continue
                
            from_id = gi.id_of(i, t, l)
            to_id = gi.id_of(j, t2, l - de)
            
            rows.append(("reposition", from_id, to_id, i, j, t, int(l), tau, int(de), dist_km))
    
    if not rows:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","dist_km"])
    
    out = pd.DataFrame(rows, columns=["arc_type","from_node_id","to_node_id","i","j","t","l","tau","de","dist_km"])
    return _drop_self_loops(out, "reposition")

def build_charging_arcs(cfg, gi: GridIndexers, reachable_set: set,
                        t0: int = None, t_hi: int = None, B: int = None) -> pd.DataFrame:
    """
    纯网络化的 charging：
      - tochg:     (i,t,l) -> (zone_k, t+tau_to, l - de_to)
      - chg_enter: (zone_k, p, lq) -> (q_in[k,p])，tau=0
      - chg_occ:   (q_in[k,p]) -> (q_out[k,p])，唯一容量弧（cap_hint = plugs_kp；成本=充电成本-奖励 由 _06_costs 附加）
      - chg_step:  (q_out[k,p]) -> (zone_k, p+1, lq')，若为最后一步则 lq' = target_soc，否则 lq'=lq（保持 SOC）
    说明：
      - 修正：考虑去站耗电，支持多轮充电以充分补偿耗电
      - 充电轮数以min_charge_step为单位，确保至少恢复到去站前的SOC
      - SOC 提升在最后一步一次性完成；车辆通过chg_step回到网格节点后可继续流动
      - 不需要额外的出站弧，chg_step已经将车辆带回网格
    """
    best_df, nearest_map = _load_zone_station_data(cfg)
    k2zone, k2level = load_stations_mapping(cfg)
    prof = load_or_build_charging_profile(cfg)

    # 最近站过滤
    pairs = [(int(i), int(k)) for i, ks in nearest_map.items() for k in ks]
    if pairs:
        near_df = pd.DataFrame(pairs, columns=["i","k"])
        i2k = best_df.merge(near_df, on=["i","k"], how="inner")
    else:
        i2k = best_df.iloc[0:0][["i","k","dist_km","tau_steps"]].copy()
    i2k = i2k[i2k["k"].isin(k2zone.keys())].copy()

    dt = int(cfg.time_soc.dt_minutes)
    arrival_end = min(t_hi + int(B) if (t_hi is not None and B is not None) else gi.times[-1], gi.times[-1])
    soc_levels = np.array(gi.socs, dtype=int)
    soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100

    # profile: (level) -> {(from_soc, to_soc): tau_minutes}
    prof_map: Dict[int, Dict[Tuple[int, int], float]] = {}
    for level, sub in prof.groupby("level"):
        d = {(int(r["from_soc"]), int(r["to_soc"])): float(r["tau_chg_minutes"]) for _, r in sub.iterrows()}
        prof_map[int(level)] = d

    min_step = int(cfg.charge_queue.min_charge_step)
    default_plugs = int(cfg.charge_queue.default_plugs_per_station)

    cap_map = load_station_capacity_map(cfg)  # k -> 有效并发（已含 plugs×util_factor×queue_relax_factor 的口径）

    rows = []
    # 时间范围
    if t0 is not None:
        time_range = range(t0, t_hi)
    else:
        time_range = gi.times

    for _, r in i2k.iterrows():
        i, k = int(r["i"]), int(r["k"])
        dist_km = float(r.get("dist_km", 0.0))
        tau_to = int(r["tau_steps"])
        level_k = int(k2level[k])
        zone_k = int(k2zone[k])

        for t in time_range:
            t_arr = t + tau_to
            if t_arr > arrival_end:
                continue
            de_to = compute_multi_timestep_energy_consumption(cfg, dist_km, t, tau_to, soc_step, "de_per_km_tochg")

            for l in soc_levels:
                if l < de_to:
                    continue
                if (i, int(l), t) not in reachable_set:
                    continue

                # 修正：考虑去站耗电，支持多轮充电以充分补偿耗电
                soc_after_travel = int(l) - int(de_to)  # 到站后的SOC
                if soc_after_travel <= 0:
                    continue
                
                # 计算需要多少轮充电来充分补偿去站耗电
                # 策略：至少充到原始SOC，然后可以继续充电
                min_target_soc = int(l)  # 至少恢复到去站前的SOC
                max_chargeable = 100 - soc_after_travel
                
                if max_chargeable <= 0:
                    continue
                
                # 计算需要的充电轮数
                needed_charge = min_target_soc - soc_after_travel
                if needed_charge <= 0:
                    # 如果到站后SOC还够，可以选择充电或不充电
                    target_soc = soc_after_travel + min_step  # 至少充一轮
                else:
                    # 需要多轮充电来补偿耗电
                    charge_rounds = math.ceil(needed_charge / min_step)  # 向上取整到整数轮
                    target_charge = min(charge_rounds * min_step, max_chargeable)
                    target_soc = soc_after_travel + int(target_charge)

                # 查询充电时间（分钟 -> 步）——从到站 SOC 充到 target_soc
                tau_chg_min = prof_map.get(level_k, {}).get((int(soc_after_travel), int(target_soc)), None)
                if tau_chg_min is None:
                    # profile 未覆盖时可退化为线性或跳过
                    continue
                tau_chg = int(math.ceil(tau_chg_min / dt))
                if tau_chg <= 0:
                    continue

                t_end = t_arr + tau_chg
                if t_end > arrival_end:
                    continue

                # 1) 到站弧 tochg - 使用专门的充电站到达节点
                if (zone_k, int(l - de_to), t_arr) not in reachable_set:
                    continue
                from_id = gi.id_of(i, t, int(l))
                # 创建专门的充电站到达节点，避免与网格节点冲突
                chg_arrival_node = _pseudo_node_id("chg_arrival", k, t_arr, int(l - de_to))
                rows.append(("tochg", from_id, chg_arrival_node, i, k, t, int(l), int(l - de_to), tau_to, tau_to, 0, int(de_to), level_k, None, None))

                # 2) 占用链：p = t_arr ... t_end-1
                soc_now = soc_after_travel  # 使用修正后的到站SOC
                p = t_arr
                for step in range(tau_chg):
                    q_in  = _pseudo_node_id("q_in",  k, p)
                    q_out = _pseudo_node_id("q_out", k, p)

                    # 2.1 chg_enter： 从专门的充电站到达节点进入充电站队列
                    # 这确保车辆必须通过tochg弧才能进入充电站
                    if (zone_k, soc_now, p) not in reachable_set:
                        break
                    # 只有在第一步(p = t_arr)时从充电站到达节点进入
                    if step == 0 and p == t_arr:
                        rows.append(("chg_enter", chg_arrival_node, q_in, i, k, p, soc_now, soc_now, 0, 0, None, None, level_k, None, None))
                    else:
                        # 后续步骤从网格节点进入（充电过程中的等待）
                        from_id = gi.id_of(zone_k, p, soc_now)
                        rows.append(("chg_enter", from_id, q_in, i, k, p, soc_now, soc_now, 0, 0, None, None, level_k, None, None))

                    # 2.2 chg_occ： q_in[k,p] -> q_out[k,p]  （唯一容量弧；cap_hint=plugs_kp）
                    plugs_kp = int(cap_map.get(k, default_plugs))  # ← 用站点并发替换默认值
                    rows.append(("chg_occ", q_in, q_out, i, k, p, soc_now, soc_now, 0, 0, None, None, level_k, plugs_kp, None))

                    # 2.3 chg_step： q_out[k,p] -> (zone_k, p+1, soc_next)
                    is_last = (step == tau_chg - 1)
                    soc_next = int(target_soc) if is_last else int(soc_now)
                    if (zone_k, soc_next, p+1) not in reachable_set:
                        break
                    to_id = gi.id_of(zone_k, p+1, soc_next)
                    rows.append(("chg_step", q_out, to_id, i, k, p, soc_now, soc_next, 1, 0, None, None, level_k, None, int(is_last)))
                    # 更新
                    soc_now = soc_next
                    p += 1
                
                # 注意：不需要额外的出站弧！
                # chg_step弧已经将车辆从充电站队列带回到网格节点(zone_k, p+1, soc_next)
                # 车辆可以从网格节点继续通过idle、reposition、service等弧流动

    if not rows:
        return pd.DataFrame(columns=[
            "arc_type","from_node_id","to_node_id","i","k","t","l","l_to",
            "tau_total","tau_tochg","tau_chg","de_tochg","level","cap_hint","is_last_step"
        ])

    out = pd.DataFrame(rows, columns=[
        "arc_type","from_node_id","to_node_id","i","k","t","l","l_to",
        "tau_total","tau_tochg","tau_chg","de_tochg","level","cap_hint","is_last_step"
    ])
    return _drop_self_loops(out, "charging")

# ============================================================
# ============ 2) "窗口动态弧生成"版（使用统一函数） ============
# ============================================================
def generate_arcs_for_window(t0: int, H: int,
                             cfg=None, gi: GridIndexers | None = None,
                             reachable_set: set | None = None) -> pd.DataFrame:
    """
    基于 Halo+承诺账本的“窗口动态弧生成”：只生成出发在窗内的弧（允许到达跨窗）。
    返回统一弧表（含 arc_type 等所有业务列，且带稳定 arc_id）。

    - 出发窗口：t ∈ [t0, t_hi-1]，其中 t_hi = min(end_step, t0 + H)
    - 到达上限：t_arr ≤ min(t_hi + B, 全局 max_t)
      其中 B = cfg.time_soc.overhang_steps

    新增/变更的 arc_type：
      idle:        ["idle"]
      service:     ["svc_enter","svc_gate","svc_exit"]
      reposition:  ["reposition"]
      charging:    ["tochg","chg_enter","chg_occ","chg_step"]
    """
    cfg = cfg or get_config()
    if gi is None:
        gi = load_indexer()
    if reachable_set is None:
        reachable_set = load_reachability_with_time()

    t_hi = min(int(cfg.time_soc.end_step), int(t0 + H))
    # Halo
    B = int(cfg.time_soc.overhang_steps)

    # 四类分别生成
    idle = build_idle_arcs(cfg, gi, reachable_set, t0, t_hi, B)
    svc  = build_service_arcs(cfg, gi, reachable_set, t0, t_hi, B)
    rep  = build_reposition_arcs(cfg, gi, reachable_set, t0, t_hi, B)
    chg  = build_charging_arcs(cfg, gi, reachable_set, t0, t_hi, B)

    # 补 arc_id（稳定键）
    idle = _attach_arc_id(idle, ["arc_type","from_node_id","to_node_id"])
    svc  = _attach_arc_id(svc,  ["arc_type","from_node_id","to_node_id","i","j","t"]) if not svc.empty else svc
    rep  = _attach_arc_id(rep,  ["arc_type","from_node_id","to_node_id"])
    chg  = _attach_arc_id(chg,  ["arc_type","from_node_id","to_node_id","k"]) if (not chg.empty and "k" in chg.columns) else chg

    frames = [df for df in (idle, svc, rep, chg) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["arc_type","from_node_id","to_node_id","t"])
    arc_df_win = pd.concat(frames, ignore_index=True)

    # 去重 & 统一兜底删除自环（防万一）
    if "arc_id" in arc_df_win.columns:
        arc_df_win = arc_df_win.drop_duplicates(subset=["arc_id"]).sort_values("arc_id").reset_index(drop=True)
    arc_df_win = _drop_self_loops(arc_df_win, "all")

    return arc_df_win

# ============================================================
# ====================== 3) 批量静态预生成 =====================
# ============================================================
def _prune_reposition_arcs_knn(rep: pd.DataFrame, K: int) -> tuple[pd.DataFrame, str, int]:
    """仅按 dist_km 升序做 KNN 剪枝；每个 from_node_id 仅保留 K 条。"""
    if rep.empty or not K or K <= 0:
        return rep, "none", 0
    key = None
    if "dist_km" in rep.columns and rep["dist_km"].notna().any() and rep["dist_km"].sum() > 0:
        key = "dist_km"
    elif "tau" in rep.columns:
        key = "tau"
    else:
        return rep, "none", 0
    rep = rep.sort_values(["from_node_id", key], kind="mergesort")
    pruned = rep.groupby("from_node_id", group_keys=False).head(int(K))
    dropped = len(rep) - len(pruned)
    return pruned, key, dropped

def main():
    """静态全量预生成（保持原有 CLI 用法）；动态滚动时请调用 generate_arcs_for_window。"""
    cfg = get_config()
    gi: GridIndexers = load_indexer()
    reachable_set = load_reachability_with_time()

    out_dir = Path("data/intermediate")
    ensure_dir(out_dir)

    # Idle arcs
    idle_df = build_idle_arcs(cfg, gi, reachable_set)
    idle_df = _drop_self_loops(idle_df, "idle")  # 再兜底
    idle_df = _attach_arc_id(idle_df, ["arc_type","from_node_id","to_node_id"])
    idle_df.to_parquet(out_dir / "idle_arcs.parquet", index=False)
    idle_df.to_csv(out_dir / "idle_arcs.csv", index=False)

    # Service arcs (纯网络化三段)
    svc_df = build_service_arcs(cfg, gi, reachable_set)
    svc_df = _drop_self_loops(svc_df, "service")
    svc_df = _attach_arc_id(svc_df, ["arc_type","from_node_id","to_node_id","i","j","t"])
    svc_df.to_parquet(out_dir / "service_arcs.parquet", index=False)
    svc_df.to_csv(out_dir / "service_arcs.csv", index=False)

    # Reposition arcs (生成 → KNN 剪枝 → 保存)
    rep_df = build_reposition_arcs(cfg, gi, reachable_set)
    rep_df = _drop_self_loops(rep_df, "reposition")
    
    # 统计需求驱动方法的效果
    reposition_demand = _compute_reposition_demand(cfg)
    rep_demand_info = f"，需求驱动：{len(reposition_demand):,} 需求对"
    if cfg.solver.verbose:
        print(f"[需求驱动] 重定位需求：{len(reposition_demand):,} 个OD对")
    
    K = cfg.pruning.reposition_nearest_zone_n or 16
    rep_prune_metric = "none"; rep_pruned_dropped = 0
    if K and K > 0 and not rep_df.empty:
        rep_df, rep_prune_metric, rep_pruned_dropped = _prune_reposition_arcs_knn(rep_df, K)
        if cfg.solver.verbose:
            print(f"[KNN] 重定位弧剪枝：K={K:,}，保留 {len(rep_df):,d} 条，剪掉 {rep_pruned_dropped:,d} 条。度量={rep_prune_metric}{rep_demand_info}")
    rep_df = _attach_arc_id(rep_df, ["arc_type","from_node_id","to_node_id"])
    rep_df.to_parquet(out_dir / "reposition_arcs.parquet", index=False)
    rep_df.to_csv(out_dir / "reposition_arcs.csv", index=False)

    # Charging arcs（占用链）
    chg_df = build_charging_arcs(cfg, gi, reachable_set)
    chg_df = _drop_self_loops(chg_df, "charging")
    chg_df = _attach_arc_id(chg_df, ["arc_type","from_node_id","to_node_id","k"])
    chg_df.to_parquet(out_dir / "charging_arcs.parquet", index=False)
    chg_df.to_csv(out_dir / "charging_arcs.csv", index=False)

    if cfg.solver.verbose:
        print("[arc_generators] saved:")
        print(f"  - idle:       data/intermediate/idle_arcs.parquet ({len(idle_df):,} rows)")
        print(f"  - service:    data/intermediate/service_arcs.parquet ({len(svc_df):,} rows)")
        print(f"  - reposition: data/intermediate/reposition_arcs.parquet ({len(rep_df):,} rows)")
        print(f"  - charging:   data/intermediate/charging_arcs.parquet ({len(chg_df):,} rows)")

    meta = {
        "idle_rows": len(idle_df),
        "service_rows": len(svc_df),
        "reposition_rows": len(rep_df),
        "charging_rows": len(chg_df),
        "dt_minutes": cfg.time_soc.dt_minutes,
        "min_charge_step": cfg.charge_queue.min_charge_step,
        "nearest_station_n": cfg.pruning.reposition_nearest_zone_n,
        "max_reposition_tt_minutes": cfg.pruning.max_reposition_tt,
        "min_soc_for_reposition": cfg.pruning.min_soc_for_reposition,
        "requires_station_zone_level": True,
        "uses_base_ij": True,
        "uses_base_i2k": True,
        "rep_pruned_knn_k": K,
        "rep_prune_metric": rep_prune_metric,
        "rep_pruned_dropped": rep_pruned_dropped,
        "charging_uses_occupancy_chain": True,
        "charging_cap_hint_default_plugs": cfg.charge_queue.default_plugs_per_station,
        "self_loops_removed": True,
    }
    (out_dir / "arcs_meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
