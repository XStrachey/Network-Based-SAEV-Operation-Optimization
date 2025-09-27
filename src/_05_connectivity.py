# _05_connectivity.py
# 仅提供“连通性裁剪 + reachability 端点过滤”的函数库（无任何全局预裁剪/落盘）
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ============================================
# 常量：兼容细分后的 arc_type（纯网络化）
# ============================================
IDLE_TYPES        = {"idle"}
SERVICE_TYPES     = {"svc_enter", "svc_gate", "svc_exit"}
REPOSITION_TYPES  = {"reposition"}
CHARGING_TYPES    = {"tochg", "chg_enter", "chg_occ", "chg_step"}

ALL_KNOWN_TYPES = IDLE_TYPES | SERVICE_TYPES | REPOSITION_TYPES | CHARGING_TYPES


# =========================
# 基础 I/O 工具（可选）
# =========================

def load_nodes(path: str | Path = "data/intermediate/nodes.parquet") -> pd.DataFrame:
    """读取 nodes（必须含 node_id, zone, t, soc）"""
    return pd.read_parquet(str(path), columns=["node_id", "zone", "t", "soc"])


def load_reachability_set(path: str | Path = "data/intermediate/reachability.parquet") -> Set[Tuple[int, int, int]]:
    """将 reachability 表转为集合 {(zone, soc, t)}"""
    df = pd.read_parquet(str(path), columns=["t", "zone", "soc", "reachable"])
    df = df[df["reachable"] == 1]
    return set((int(r.zone), int(r.soc), int(r.t)) for r in df.itertuples())


# 使用 data_loader 中的共用函数
from data_loader import load_initial_inventory_parquet as load_initial_inventory


# =========================
# 连通性（FWD/BWD）与裁剪
# =========================

def _bfs(starts: Iterable[int], adj: Dict[int, List[int]]) -> Set[int]:
    """
    标准 BFS（节点集合为整数 node_id）
    说明：支持“伪节点”（如 svc_in/out、q_in/out）——它们可能为负数，也无需出现在 nodes 表中，
         只要它们作为弧端点出现在邻接表里，BFS 即可遍历。
    """
    seen: Set[int] = set()
    dq = deque(int(s) for s in starts)
    for s in dq:
        seen.add(s)
    while dq:
        u = dq.popleft()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen


def _build_adjacency(arcs_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """由弧构造正向/反向邻接表（遍历传入字典中的所有弧表）"""
    out_adj: Dict[int, List[int]] = defaultdict(list)
    in_adj: Dict[int, List[int]] = defaultdict(list)
    for k, df in arcs_dict.items():
        if df is None or df.empty:
            continue
        need = {"from_node_id", "to_node_id"}
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"[connectivity] {k} 缺列: {miss}")
        # 忽略含 NaN 的端点行（极少见；一般不应出现）
        sub = df[["from_node_id", "to_node_id"]].dropna()
        for u, v in sub.itertuples(index=False):
            uu = int(u)
            vv = int(v)
            out_adj[uu].append(vv)
            in_adj[vv].append(uu)
    return out_adj, in_adj


@dataclass
class ConnResult:
    FWD: Set[int]
    BWD: Set[int]
    KEEP: Set[int]
    arcs_after: Dict[str, pd.DataFrame]


def compute_connectivity_keep_sets(
    arcs_dict: Dict[str, pd.DataFrame],
    start_nodes: Set[int],
    final_nodes: Optional[Set[int]] = None,
    require_bwd: bool = False,
    keep_on: str = "from",
) -> ConnResult:
    """
    计算 FWD / BWD / KEEP，并按 KEEP 裁剪弧（仅删行）。

    参数
    - start_nodes：作为“源”起点集合（建议：t==t0 的供给节点 + 窗口内所有外生入流的 to_node_id）
    - final_nodes：作为“终端”集合（仅在 require_bwd=True 时使用）
    - require_bwd：True -> 需要能到终端；False -> 仅检查前向可达（默认）
    - keep_on：'both' | 'from' | 'either'
        * 'from'（默认）：保留 from_node ∈ KEEP 的弧（适配跨窗到达的弧）
        * 'both'：保留 from_node、to_node 都 ∈ KEEP（传统双端裁剪）
        * 'either'：任一端落在 KEEP 即保留（最宽松）

    兼容事项
    - 支持新 arc_type：svc_enter/gate/exit, tochg/chg_enter/chg_occ/chg_step 等；
      这些弧的端点可为负的伪节点（闸门节点或充电队列节点），BFS 会自然覆盖。
    """
    out_adj, in_adj = _build_adjacency(arcs_dict)
    FWD = _bfs(start_nodes, out_adj) if start_nodes else set()
    if require_bwd and final_nodes and len(final_nodes) > 0:
        BWD = _bfs(final_nodes, in_adj)
        KEEP = FWD & BWD
    else:
        BWD = set()
        KEEP = FWD

    def _prune(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not KEEP:
            return df.iloc[0:0].copy()
        if keep_on == "both":
            m = df["from_node_id"].isin(KEEP) & df["to_node_id"].isin(KEEP)
        elif keep_on == "either":
            m = df["from_node_id"].isin(KEEP) | df["to_node_id"].isin(KEEP)
        else:  # 'from'
            m = df["from_node_id"].isin(KEEP)
        return df[m.values].copy()

    arcs_after = {k: _prune(df) for k, df in arcs_dict.items()}
    return ConnResult(FWD=FWD, BWD=BWD, KEEP=KEEP, arcs_after=arcs_after)


# =========================
# 面向“窗口”的一站式工具
# =========================

def split_arcs_by_type(arc_df: pd.DataFrame, atype_col: str = "arc_type") -> Dict[str, pd.DataFrame]:
    """
    将统一弧表拆为 {idle, service, reposition, charging[, other]} 字典。
    - 保留所有原列（便于后续 LP 使用）
    - 默认类型列为 'arc_type'
    - 兼容纯网络化后的细分类型（svc_*, chg_*）
    """
    if arc_df is None or arc_df.empty:
        empty = arc_df.iloc[0:0] if arc_df is not None else pd.DataFrame()
        return {"idle": empty, "service": empty, "reposition": empty, "charging": empty, "other": empty}

    req = {"from_node_id", "to_node_id", atype_col}
    miss = [c for c in req if c not in arc_df.columns]
    if miss:
        raise ValueError(f"[split] 弧表缺列: {miss}")

    def sub(types: Set[str]) -> pd.DataFrame:
        return arc_df[arc_df[atype_col].isin(types)].copy()

    arcs_dict: Dict[str, pd.DataFrame] = {
        "idle":       sub(IDLE_TYPES),
        "service":    sub(SERVICE_TYPES),
        "reposition": sub(REPOSITION_TYPES),
        "charging":   sub(CHARGING_TYPES),
    }
    # 兜底：未识别类型（如果未来扩展了 arc_type 而未同步这里）
    other = arc_df[~arc_df[atype_col].isin(ALL_KNOWN_TYPES)].copy()
    arcs_dict["other"] = other
    return arcs_dict


def merge_arcs_dict(arcs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """将多类弧重新拼回统一表（若有 arc_id 索引请在外层维护）"""
    frames = [df for df in arcs_dict.values() if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def dynamic_prune_for_window(
    arc_df_win: pd.DataFrame,
    nodes_win: pd.DataFrame,
    start_nodes: Set[int],
    final_nodes: Set[int],
    *,
    require_bwd: bool = False,
    keep_on: str = "from",
    atype_col: str = "arc_type",
) -> Tuple[pd.DataFrame, Dict[str, int], ConnResult]:
    """
    供 07 在“每个滚动窗口”调用的一站式裁剪：
      步骤：
        1) 以 start_nodes 为源做 FWD（与可选 BWD）；
        2) 按 keep_on 规则保留弧（默认 'from' 保留起点在 FWD 内的弧）；
        3) 返回裁剪后的弧（仍包含所有业务列）、统计与连通性集合。

    关键说明（配合“允许跨窗弧”的滚动策略）：
      - arc_df_win 建议按“出发时刻 t ∈ [t0, t_hi-1]”筛选，不再要求“到达 ≤ t_hi”；
      - require_bwd 默认 False（仅 FWD），不强制必须能抵达窗末；
      - keep_on 默认 'from'，只要起点可达就保留该弧（终点可在窗外/或未出现在 nodes_win 中）。
      - 支持 svc_* 与 chg_* 弧；伪节点（负 id）不需要在 nodes_win 中出现。

    参数：
      - arc_df_win：窗口内候选弧（建议只按出发时刻入窗）
      - nodes_win：窗口节点，需含 node_id, zone, t, soc（目前仅用于构造起止集合的辅助函数）
      - start_nodes：窗口内可注入流量的起点节点集合（t==t0 的供给 + 窗口内外生入流到达节点）
      - final_nodes：窗口末节点集合（若 require_bwd=True 时参与 BWD）
      - require_bwd：是否启用 BWD（False 则仅做 FWD）
      - keep_on：裁剪保留标准，见上
      - atype_col：弧类型列名（默认 'arc_type'）

    返回：
      pruned_df：裁剪后的统一弧表
      stats：计数统计（before/after/removed）
      conn：连通性集合与分类型裁剪结果
    """
    if arc_df_win is None or arc_df_win.empty:
        empty = arc_df_win.iloc[0:0].copy() if arc_df_win is not None else pd.DataFrame()
        return empty, {"before": 0, "after": 0, "removed": 0}, ConnResult(set(), set(), set(), {})

    # 1) 弧分类（reachability 已在弧生成时过滤）
    arcs_dict = split_arcs_by_type(arc_df_win, atype_col=atype_col)
    before = sum(len(v) for v in arcs_dict.values())

    # 2) FWD/BWD/KEEP + 按 KEEP 裁剪（保留规则由 keep_on 控制）
    conn = compute_connectivity_keep_sets(
        arcs_dict=arcs_dict,
        start_nodes=set(int(x) for x in start_nodes),
        final_nodes=set(int(x) for x in final_nodes) if final_nodes else None,
        require_bwd=require_bwd,
        keep_on=keep_on,
    )

    # 3) 合并
    pruned_df = merge_arcs_dict(conn.arcs_after)
    after = int(len(pruned_df))
    stats = {"before": int(before), "after": after, "removed": int(before - after)}
    return pruned_df, stats, conn


# =========================
# 辅助：窗口起点/终点节点构造
# =========================

def source_nodes_from_inventory_at_t0(V0: pd.DataFrame, nodes: pd.DataFrame, t0: int) -> Set[int]:
    """
    从初始库存构造窗口起点节点集合（仅 t==t0 的正库存）。
    V0: zone,soc,t,count；nodes: node_id, zone, t, soc
    """
    v = V0[V0["t"] == int(t0)].copy()
    if v.empty:
        return set()
    v = v[v["count"] > 0]
    jj = v.merge(nodes, on=["zone", "t", "soc"], how="left", validate="many_to_one")
    return set(jj["node_id"].dropna().astype(np.int64).tolist())


def final_nodes_at_t(nodes: pd.DataFrame, t_hi: int) -> Set[int]:
    """取 t==t_hi 的所有节点作为窗口终端"""
    sub = nodes[nodes["t"] == int(t_hi)]
    return set(sub["node_id"].astype(np.int64).tolist())
