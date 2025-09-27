# graph_ops/reposition_r1_pruner.py
# 实现 "R1｜需求压力梯度导向" 的重定位弧定向裁剪

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, Optional, Any
import random
from collections import defaultdict
import networkx as nx

def prune_reposition_arcs(
    arcs_df: pd.DataFrame,            # 全部弧（DataFrame）
    nodes_df: pd.DataFrame,           # 节点（含 zone i、time t、soc l）
    svc_gates_df: pd.DataFrame,       # 仅 type=='svc_gate' 的弧（含 i,j,t 与容量/需求）
    prev_solution_df: Optional[pd.DataFrame] = None,    # 可选：上一窗口（或粗解）流量，用于供给估计
    cfg: Optional[Dict[str, Any]] = None,            # R1_PRUNE 配置
    overhang_steps: Optional[int] = None             # 跨窗步数限制，来自 time_soc.overhang_steps
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    返回: pruned_arcs_df, report(dict)
    关键列假设:
      - arcs_df: ['arc_id','arc_type','from_node_id','to_node_id','cost','tau','i','j','t','l', ...]
      - svc_gates_df: ['i','j','t','cap_hint'] (cap_hint = demand)
      - nodes_df: ['node_id','zone','t', 'soc']
    """
    if cfg is None:
        cfg = {}
    
    # 默认配置
    default_cfg = {
        'enabled': True,
        'delta': 0.0,
        'epsilon': 0.05,
        'keep_reverse_ratio': 0.10,
        'min_outdeg': 2,
        'min_indeg': 2,
        'supply_mode': 'none',
        'random_seed': 13
    }
    
    # 合并配置
    for key, default_val in default_cfg.items():
        if key not in cfg:
            cfg[key] = default_val
    
    if not cfg['enabled']:
        return arcs_df, {'reposition_before': 0, 'reposition_after': 0, 'drop_ratio': 0.0, 
                        'kept_reverse': 0, 'min_outdeg_violation': 0, 'min_indeg_violation': 0, 
                        'recovered_edges': 0, 'halo_filtered': 0}
    
    # 设置随机种子
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    
    # 分离重定位弧和其他弧
    reposition_mask = arcs_df['arc_type'] == 'reposition'
    repos_df = arcs_df[reposition_mask].copy()
    other_arcs_df = arcs_df[~reposition_mask].copy()
    
    if repos_df.empty:
        return arcs_df, {'reposition_before': 0, 'reposition_after': 0, 'drop_ratio': 0.0, 
                        'kept_reverse': 0, 'min_outdeg_violation': 0, 'min_indeg_violation': 0, 
                        'recovered_edges': 0, 'halo_filtered': 0}
    
    initial_count = len(repos_df)
    
    # 1) 计算需求压力 DP(i,t)
    dp_map = _compute_demand_pressure(svc_gates_df, prev_solution_df, cfg)
    
    # 2) 对重定位弧计算 Δ 并定向
    repos_df = _compute_delta_dp(repos_df, dp_map)
    
    # 3) 应用 Halo 过滤
    # 使用 overhang_steps 作为跨窗步数限制
    max_halo_steps = overhang_steps if overhang_steps is not None else 2  # 默认值
    repos_df, halo_filtered = _apply_halo_filter(repos_df, max_halo_steps)
    
    # 4) 定向裁剪
    repos_df = _apply_directional_pruning(repos_df, cfg)
    
    # 5) 兜底与连通性健康检查
    repos_df, safety_report = _apply_safety_checks(repos_df, nodes_df, cfg)
    
    # 6) 合并结果
    final_arcs_df = pd.concat([other_arcs_df, repos_df], ignore_index=True)
    
    # 7) 生成报告
    report = {
        'reposition_before': initial_count,
        'reposition_after': len(repos_df),
        'drop_ratio': (initial_count - len(repos_df)) / initial_count if initial_count > 0 else 0.0,
        'kept_reverse': safety_report.get('kept_reverse', 0),
        'min_outdeg_violation': safety_report.get('min_outdeg_violation', 0),
        'min_indeg_violation': safety_report.get('min_indeg_violation', 0),
        'recovered_edges': safety_report.get('recovered_edges', 0),
        'halo_filtered': halo_filtered
    }
    
    return final_arcs_df, report


def _compute_demand_pressure(
    svc_gates_df: pd.DataFrame, 
    prev_solution_df: Optional[pd.DataFrame], 
    cfg: Dict[str, Any]
) -> Dict[Tuple[int, int], float]:
    """
    计算需求压力 DP(i,t) = D_i(t) - S_i(t)
    返回: {(zone, time): DP_value}
    """
    # 需求 D_i(t) = Σ_j D_ij(t)
    demand = (svc_gates_df
              .groupby(['i', 't'], as_index=False)
              .agg(D=('cap_hint', 'sum')))
    
    # 供给 S_i(t)：阶段1默认不用（supply_mode='none'）
    if cfg['supply_mode'] == 'none':
        supply = demand[['i', 't']].copy()
        supply['S'] = 0.0
    else:
        # 预留：从 prev_solution_df 估计
        supply = _estimate_supply(prev_solution_df, demand, cfg)
    
    dp = (demand
          .merge(supply, on=['i', 't'], how='left')
          .rename(columns={'i': 'zone'}))
    dp['S'] = dp['S'].fillna(0.0)
    dp['DP'] = dp['D'] - dp['S']
    
    # 快速索引 (zone, t) -> DP 值
    dp_map = {(int(r.zone), int(r.t)): float(r.DP) for r in dp.itertuples(index=False)}
    
    return dp_map


def _estimate_supply(
    prev_solution_df: Optional[pd.DataFrame], 
    demand: pd.DataFrame, 
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    预留：从上一窗口解估计供给 S_i(t)
    """
    # 简化实现：返回0供给
    supply = demand[['i', 't']].copy()
    supply['S'] = 0.0
    return supply


def _compute_delta_dp(
    repos_df: pd.DataFrame, 
    dp_map: Dict[Tuple[int, int], float]
) -> pd.DataFrame:
    """
    对重定位弧计算 Δ = DP(j, t_to) - DP(i, t_from)
    """
    def dp_lookup(zone, t):
        return dp_map.get((int(zone), int(t)), 0.0)
    
    # 计算 t_to (到达时间)
    repos_df['t_to'] = repos_df['t'] + repos_df['tau']
    
    # 计算 Δ = DP(j, t_to) - DP(i, t_from)
    repos_df['DP_from'] = [dp_lookup(z, t) for z, t in zip(repos_df['i'], repos_df['t'])]
    repos_df['DP_to'] = [dp_lookup(z, t) for z, t in zip(repos_df['j'], repos_df['t_to'])]
    repos_df['delta_dp'] = repos_df['DP_to'] - repos_df['DP_from']
    
    return repos_df


def _apply_halo_filter(
    repos_df: pd.DataFrame, 
    max_halo_steps: int
) -> Tuple[pd.DataFrame, int]:
    """
    应用时间/窗口护栏：若 t_to - t_from > max_halo_steps 则删除该边
    
    注意：max_halo_steps 现在直接使用 time_soc.overhang_steps，避免重复配置
    """
    initial_count = len(repos_df)
    
    # 过滤超出 Halo 范围的弧
    repos_df = repos_df[repos_df['tau'] <= max_halo_steps].copy()
    
    halo_filtered = initial_count - len(repos_df)
    return repos_df, halo_filtered


def _apply_directional_pruning(
    repos_df: pd.DataFrame, 
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    应用定向裁剪规则
    """
    delta = cfg['delta']
    epsilon = cfg['epsilon']
    
    # 构造无向键用于成对处理
    repos_df['key'] = repos_df.apply(
        lambda row: (min(row['i'], row['j']), max(row['i'], row['j']), row['t']), 
        axis=1
    )
    
    # 按键分组处理
    kept_arcs = []
    for key, group in repos_df.groupby('key'):
        kept_arcs.extend(_process_symmetric_pair(group, delta, epsilon, cfg))
    
    if kept_arcs:
        result_df = pd.concat(kept_arcs, ignore_index=True)
        # 清理临时列
        result_df = result_df.drop(columns=['key', 't_to', 'DP_from', 'DP_to', 'delta_dp'], errors='ignore')
    else:
        result_df = repos_df.iloc[0:0].copy()  # 空DataFrame，保持列结构
    
    return result_df


def _process_symmetric_pair(
    group: pd.DataFrame, 
    delta: float, 
    epsilon: float, 
    cfg: Dict[str, Any]
) -> list[pd.DataFrame]:
    """
    处理一对对称重定位邻居的定向决策
    """
    if len(group) <= 1:
        return [group]
    
    # 分离两个方向
    i_to_j = group[group['i'] < group['j']]
    j_to_i = group[group['i'] > group['j']]
    
    if i_to_j.empty or j_to_i.empty:
        # 只有一个方向，保留
        return [group]
    
    # 获取代表弧的 delta_dp
    i_to_j_delta = i_to_j.iloc[0]['delta_dp']
    j_to_i_delta = j_to_i.iloc[0]['delta_dp']
    
    kept_groups = []
    
    # 判定规则
    if abs(i_to_j_delta) <= epsilon:
        # 不确定带：两向都保留
        kept_groups.extend([i_to_j, j_to_i])
    elif i_to_j_delta >= delta:
        # 保留 i→j，候选删 j→i
        kept_groups.append(i_to_j)
        # 按成本/距离在 j→i 保留少量兜底
        kept_groups.append(_keep_reverse_ratio(j_to_i, cfg['keep_reverse_ratio']))
    elif j_to_i_delta >= delta:
        # 保留 j→i，候选删 i→j
        kept_groups.append(j_to_i)
        # 按成本/距离在 i→j 保留少量兜底
        kept_groups.append(_keep_reverse_ratio(i_to_j, cfg['keep_reverse_ratio']))
    else:
        # 都不满足阈值，保留两向
        kept_groups.extend([i_to_j, j_to_i])
    
    return kept_groups


def _keep_reverse_ratio(group: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    按成本/距离保留指定比例的反向边
    """
    if group.empty or ratio <= 0:
        return group.iloc[0:0].copy()  # 返回空DataFrame
    
    # 按成本排序（如果存在cost列）
    if 'cost' in group.columns:
        group = group.sort_values('cost')
    elif 'dist_km' in group.columns:
        group = group.sort_values('dist_km')
    
    # 保留前ratio比例的边
    keep_count = max(1, int(len(group) * ratio))
    return group.head(keep_count)


def _apply_safety_checks(
    repos_df: pd.DataFrame, 
    nodes_df: pd.DataFrame, 
    cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    应用兜底与连通性健康检查
    """
    safety_report = {
        'kept_reverse': 0,
        'min_outdeg_violation': 0,
        'min_indeg_violation': 0,
        'recovered_edges': 0
    }
    
    if repos_df.empty:
        return repos_df, safety_report
    
    # 兜底1：最小度检查
    repos_df, deg_violations = _ensure_min_degree(repos_df, cfg)
    safety_report['min_outdeg_violation'] = deg_violations['outdeg']
    safety_report['min_indeg_violation'] = deg_violations['indeg']
    
    # 兜底2：连通性检查（简化版）
    # 在实际实现中，这里应该进行更复杂的连通性检查
    # 目前先跳过，因为连通性检查需要完整的图结构
    
    return repos_df, safety_report


def _ensure_min_degree(
    repos_df: pd.DataFrame, 
    cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    确保每个 (zone, time) 的最小出入度
    """
    min_outdeg = cfg['min_outdeg']
    min_indeg = cfg['min_indeg']
    
    violations = {'outdeg': 0, 'indeg': 0}
    
    # 计算当前出入度
    outdeg = repos_df.groupby(['i', 't']).size().to_dict()
    indeg = repos_df.groupby(['j', 't']).size().to_dict()
    
    # 检查出度违规
    outdeg_violations = []
    for (zone, t), count in outdeg.items():
        if count < min_outdeg:
            outdeg_violations.append((zone, t))
            violations['outdeg'] += min_outdeg - count
    
    # 检查入度违规
    indeg_violations = []
    for (zone, t), count in indeg.items():
        if count < min_indeg:
            indeg_violations.append((zone, t))
            violations['indeg'] += min_indeg - count
    
    # 简化实现：暂时不进行回补，仅记录违规数量
    # 在实际实现中，应该从候选删除集中按最小成本回补边
    
    return repos_df, violations


def _check_connectivity(
    repos_df: pd.DataFrame, 
    nodes_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    检查连通性（预留实现）
    """
    # 简化实现：返回基本统计
    return {
        'connected_components': 1,
        'isolated_nodes': 0,
        'connectivity_passed': True
    }
