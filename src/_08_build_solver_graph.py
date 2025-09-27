# build_solver_graph.py
# 用 02~06 的模块把“滚动窗口”的图一次性构造成独立求解器可用的输入
# 现在：--t0 / --H 为可选；缺省时从配置读取 cfg.time_soc.start_step / cfg.time_soc.window_length
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# 兼容两种配置模块路径：network_config.py 或 network/_01_network_config.py
try:
    from _01_network_config import get_network_config as get_config
except Exception:  # noqa
    from network._01_network_config import get_network_config as get_config  # type: ignore

# --- 引用 02~06 的实现 ---
from _02_build_time_soc_grid import (
    build_indexers, materialize_nodes_dataframe, load_initial_inventory, save_artifacts
)
from _03_reachability import (
    build_zone_station_best, select_nearest_stations,
    build_zone_station_energy_table, build_energy_agg, build_reachability_table
)
from _04_arc_generators import generate_arcs_for_window
from _05_connectivity import (
    dynamic_prune_for_window, source_nodes_from_inventory_at_t0, final_nodes_at_t
)
from _06_costs import attach_costs_to_arcs_for_window
from graph_ops.reposition_r1_pruner import prune_reposition_arcs


# ----------------------------
# 小工具
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _pseudo_node_id(kind: str, *keys) -> int:
    """与 04 一致的伪节点 id 生成（负数），用于超级汇点。"""
    import hashlib
    s = f"{kind}|" + "|".join(map(str, keys))
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF
    return -int(val)

def _maybe_build_grid(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """若缺失 02 的产物，则即时构建；返回 (nodes, V0)。"""
    nodes_p = Path("data/intermediate/nodes.parquet")
    inv_p   = Path("data/intermediate/initial_inventory.parquet")
    if nodes_p.exists() and inv_p.exists():
        nodes = pd.read_parquet(nodes_p)
        V0    = pd.read_parquet(inv_p)
        return nodes, V0
    gi = build_indexers(cfg)
    nodes = materialize_nodes_dataframe(gi)
    V0 = load_initial_inventory(cfg, gi)
    save_artifacts(cfg, gi, nodes, V0)
    return nodes, V0

def _maybe_build_reachability(cfg) -> None:
    """若缺失 03 的中间件，调用 03 的函数生成（而非直接 main）。"""
    out_dir = Path("data/intermediate")
    _ensure_dir(out_dir)
    best_p = out_dir / "zone_station_best.parquet"
    near_p = out_dir / "nearest_stations.json"
    reach_p = out_dir / "reachability.parquet"
    if best_p.exists() and near_p.exists() and reach_p.exists():
        return

    from grid_utils import load_indexer
    gi = load_indexer()

    best = build_zone_station_best(cfg, gi)
    best.to_parquet(best_p, index=False)

    nearest_map, prune_stats = select_nearest_stations(cfg, best)
    (out_dir/"nearest_stations.json").write_text(json.dumps({str(k): v for k,v in nearest_map.items()}, indent=2))
    prune_stats.to_csv(out_dir/"nearest_stations_pruning.csv", index=False)

    energy_kept = build_zone_station_energy_table(cfg, gi, best, nearest_map)
    energy_kept.to_csv(out_dir/"zone_station_energy_kept.csv", index=False)
    energy_agg = build_energy_agg(energy_kept)
    energy_agg.to_csv(out_dir/"zone_energy_nearest_agg.csv", index=False)

    reach, _ = build_reachability_table(cfg, gi, best, nearest_map)
    reach.to_parquet(reach_p, index=False)
    reach.to_csv(out_dir/"reachability.csv", index=False)

def _compute_supply_vector_at_t0(V0: pd.DataFrame, nodes: pd.DataFrame, t0: int) -> pd.DataFrame:
    """返回 nodes[['node_id','zone','t','soc','supply']]，仅 t==t0 节点有正供给。"""
    sub = V0[(V0["t"] == int(t0)) & (V0["count"] > 0)].copy()
    nodes_out = nodes.copy()
    nodes_out["supply"] = 0.0
    if sub.empty:
        return nodes_out[["node_id","zone","t","soc","supply"]]
    jj = sub.merge(nodes, on=["zone","t","soc"], how="left", validate="many_to_one")
    if jj["node_id"].isna().all():
        return nodes_out[["node_id","zone","t","soc","supply"]]
    g = jj.dropna(subset=["node_id"]).groupby("node_id", as_index=False)["count"].sum()
    mp = dict(zip(g["node_id"].astype(np.int64), g["count"].astype(float)))
    sel = nodes_out["node_id"].map(mp)
    nodes_out.loc[sel.notna(), "supply"] = sel.dropna().astype(float)
    return nodes_out[["node_id","zone","t","soc","supply"]]

def _add_super_sink_arcs(arcs: pd.DataFrame, final_nodes: Set[int], nodes: pd.DataFrame, 
                        cost_to_sink: float = 0.0, cap_inf: float = 1e12, 
                        end_soc_config=None) -> Tuple[pd.DataFrame, int]:
    """为 t==t_hi 的所有节点添加到 super sink 的弧，支持期末SOC约束。"""
    if not final_nodes:
        return arcs, 0
    
    sink_id = _pseudo_node_id("super_sink")
    
    # 构建基础弧数据
    add_df = pd.DataFrame({
        "arc_type": "to_sink",
        "from_node_id": list(map(int, final_nodes)),
        "to_node_id": int(sink_id),
        "cost": float(cost_to_sink),
        "capacity": float(cap_inf),
    })
    
    # 应用期末SOC约束
    if end_soc_config is not None:
        # 获取节点信息以计算SOC
        nodes_dict = nodes.set_index("node_id").to_dict("index")
        
        def compute_soc_penalty(node_id):
            if node_id not in nodes_dict:
                return 0.0
            
            node_info = nodes_dict[node_id]
            soc = node_info.get("soc", 0)
            
            # 硬约束：如果设置了end_soc_min，则低于阈值的节点不连to_sink
            if end_soc_config.end_soc_min is not None and soc < end_soc_config.end_soc_min:
                return float('inf')  # 用无穷大成本阻止连接
            
            # 软约束：计算SOC惩罚
            if end_soc_config.end_soc_penalty_per_pct > 0 and end_soc_config.end_soc_min is not None:
                penalty = max(0, end_soc_config.end_soc_min - soc) * end_soc_config.end_soc_penalty_per_pct
                return float(cost_to_sink) + penalty
            
            return float(cost_to_sink)
        
        add_df["cost"] = add_df["from_node_id"].apply(compute_soc_penalty)
        
        # 过滤掉无穷大成本的弧（硬约束）
        add_df = add_df[add_df["cost"] != float('inf')].copy()
    
    if add_df.empty:
        return arcs, 0
    
    add_df["arc_id"] = pd.util.hash_pandas_object(
        add_df[["arc_type","from_node_id","to_node_id"]].astype(str).agg("|".join, axis=1), index=False
    ).astype("int64") & np.int64(0x7FFFFFFFFFFFFFFF)
    
    # 对齐字段
    for c in arcs.columns:
        if c not in add_df.columns:
            add_df[c] = pd.NA
    
    arcs_out = pd.concat([arcs, add_df[arcs.columns]], ignore_index=True)
    return arcs_out, int(sink_id)


def _validate_end_soc_ratio(V0: pd.DataFrame, end_inventory: pd.DataFrame, min_ratio: float = 1.0) -> Dict[str, float]:
    """验证期末总能量比是否满足阈值要求。"""
    # 计算期初总能量
    E0 = float((V0["supply"] * V0["soc"]).sum())
    
    # 计算期末总能量
    E_T = float((end_inventory["supply"] * end_inventory["soc"]).sum())
    
    # 计算比例
    ratio = E_T / E0 if E0 > 0 else 0.0
    
    return {
        "E0": E0,
        "E_T": E_T,
        "ratio": ratio,
        "min_ratio": min_ratio,
        "passed": ratio >= min_ratio
    }


# ----------------------------
# 主流程
# ----------------------------
def build_solver_graph(
    t0: Optional[int] = None,
    H: Optional[int] = None,
    keep_on: str = "from",
    require_bwd: bool = False,
    add_sink: bool = True,
    cost_to_sink: float = 0.0,
    cap_infinite: float = 1e12,
    out_dir: str = "data/solver_graph",
) -> Dict[str, str]:
    """
    返回落盘路径字典。t0/H 缺省时从配置读取。
    """
    cfg = get_config()
    _ensure_dir(Path(out_dir))

    # 读取配置默认
    if t0 is None:
        t0 = int(cfg.time_soc.start_step)
    if H is None:
        H = int(cfg.time_soc.window_length or 0)
        if H <= 0:
            # 兜底：若未配置 window_length，则用 end_step - t0 + 1
            H = int(cfg.time_soc.end_step) - int(t0) + 1

    # 0) 保障 02 / 03 产物
    nodes, V0 = _maybe_build_grid(cfg)
    _maybe_build_reachability(cfg)

    # 1) 弧：窗口生成（04）
    arc_df_win = generate_arcs_for_window(t0=int(t0), H=int(H))

    # 2) 连通性裁剪（05）
    from grid_utils import load_indexer
    gi = load_indexer()
    t_hi = min(int(cfg.time_soc.end_step), int(t0 + H))
    nodes_win = nodes[(nodes["t"] >= int(t0)) & (nodes["t"] <= int(t_hi))].copy()

    start_nodes = source_nodes_from_inventory_at_t0(V0, nodes, t0=int(t0))
    final_nodes = final_nodes_at_t(nodes, t_hi=int(t_hi))

    pruned_df, stats, _conn = dynamic_prune_for_window(
        arc_df_win=arc_df_win,
        nodes_win=nodes_win,
        start_nodes=start_nodes,
        final_nodes=final_nodes,
        require_bwd=bool(require_bwd),
        keep_on=str(keep_on),
        atype_col="arc_type",
    )

    # 3) 附加成本（06）
    arcs_costed = attach_costs_to_arcs_for_window(pruned_df)

    # 4) R1重定位弧定向裁剪
    if cfg.r1_prune.enabled:
        print("[solver-graph] 执行R1重定位弧定向裁剪...")
        svc_gates_df = arcs_costed[arcs_costed['arc_type'] == 'svc_gate'].copy()
        if not svc_gates_df.empty:
            arcs_costed, r1_report = prune_reposition_arcs(
                arcs_df=arcs_costed,
                nodes_df=nodes_win,
                svc_gates_df=svc_gates_df,
                prev_solution_df=None,   # 初版先设 None；后续可接入上一窗口解
                cfg=cfg.r1_prune.as_dict() if hasattr(cfg.r1_prune, 'as_dict') else {
                    'enabled': cfg.r1_prune.enabled,
                    'delta': cfg.r1_prune.delta,
                    'epsilon': cfg.r1_prune.epsilon,
                    'keep_reverse_ratio': cfg.r1_prune.keep_reverse_ratio,
                    'min_outdeg': cfg.r1_prune.min_outdeg,
                    'min_indeg': cfg.r1_prune.min_indeg,
                    'supply_mode': cfg.r1_prune.supply_mode,
                    'random_seed': cfg.r1_prune.random_seed
                },
                overhang_steps=int(cfg.time_soc.overhang_steps)  # 直接使用 time_soc.overhang_steps
            )
            print(f"[R1_PRUNE] 重定位弧数量变化: {r1_report['reposition_before']:,} -> {r1_report['reposition_after']:,}")
            print(f"[R1_PRUNE] 裁剪比例: {r1_report['drop_ratio']:.1%}")
            print(f"[R1_PRUNE] Halo过滤: {r1_report['halo_filtered']:,} 条弧")
            if r1_report['min_outdeg_violation'] > 0 or r1_report['min_indeg_violation'] > 0:
                print(f"[R1_PRUNE] 度违规: 出度{r1_report['min_outdeg_violation']}, 入度{r1_report['min_indeg_violation']}")
        else:
            print("[R1_PRUNE] 跳过：未找到服务闸门弧")
    else:
        print("[solver-graph] R1重定位弧定向裁剪已禁用")

    # 5) 容量字段：svc_gate 与 chg_occ 使用 cap_hint，其它给"无穷容量"
    arcs_costed = arcs_costed.copy()
    if "capacity" not in arcs_costed.columns:
        arcs_costed["capacity"] = np.nan

    mask_gate = arcs_costed["arc_type"].eq("svc_gate")
    mask_occ  = arcs_costed["arc_type"].eq("chg_occ")
    cap_hint = arcs_costed.get("cap_hint", pd.Series(index=arcs_costed.index, dtype="float64")).fillna(0.0)

    arcs_costed.loc[mask_gate | mask_occ, "capacity"] = cap_hint[mask_gate | mask_occ].astype(float)
    arcs_costed.loc[~(mask_gate | mask_occ), "capacity"] = float(cap_infinite)

    # 6) 单一成本列：cost = coef_total
    if "cost" not in arcs_costed.columns:
        arcs_costed["cost"] = 0.0
    arcs_costed["cost"] = arcs_costed["coef_total"].fillna(0.0).astype(float)

    # 7) 可选：添加超级汇点
    sink_id = None
    if add_sink:
        arcs_costed, sink_id = _add_super_sink_arcs(
            arcs_costed, 
            final_nodes=final_nodes, 
            nodes=nodes,
            cost_to_sink=float(cost_to_sink), 
            cap_inf=float(cap_infinite),
            end_soc_config=cfg.end_soc
        )

    # 8) 节点供给
    nodes_with_supply = _compute_supply_vector_at_t0(V0, nodes, t0=int(t0))
    if add_sink:
        total_sup = float(nodes_with_supply["supply"].sum())
        sink_row = pd.DataFrame({"node_id":[int(sink_id)], "zone":[pd.NA], "t":[pd.NA], "soc":[pd.NA], "supply":[-total_sup]})
        nodes_with_supply = pd.concat([nodes_with_supply, sink_row], ignore_index=True)

    # >>> 新增：补齐伪节点（svc_in/svc_out/q_in/q_out 等），设为 0 供给 <<<
    all_arc_nodes = set(arcs_costed["from_node_id"].dropna().astype(np.int64)) \
                | set(arcs_costed["to_node_id"].dropna().astype(np.int64))
    known_nodes = set(nodes_with_supply["node_id"].dropna().astype(np.int64))
    missing = sorted(all_arc_nodes - known_nodes)
    if missing:
        # 保留超级汇点的供给设置，其他伪节点设为0供给
        pseudo_supplies = []
        for node_id in missing:
            if node_id == sink_id:
                # 保持超级汇点的负供给
                pseudo_supplies.append(-total_sup)
            else:
                # 其他伪节点设为0供给
                pseudo_supplies.append(0.0)
        
        pseudo_rows = pd.DataFrame({
            "node_id": missing,
            "zone": pd.NA, "t": pd.NA, "soc": pd.NA,
            "supply": pseudo_supplies
        })
        nodes_with_supply = pd.concat([nodes_with_supply, pseudo_rows], ignore_index=True)

    # 9) 仅保留弧中实际出现的节点
    used_nodes = set(arcs_costed["from_node_id"].dropna().astype(np.int64)) | set(arcs_costed["to_node_id"].dropna().astype(np.int64))
    nodes_final = nodes_with_supply[nodes_with_supply["node_id"].isin(used_nodes)].copy()

    # 10) 负环检测（可选）
    # if cfg.solver.check_negative_cycles:
    #     from _07_sanity import check_negative_cycles, export_negative_cycle_report
        
    #     print("[solver-graph] 执行负环检测...")
    #     sanity_result = check_negative_cycles(
    #         arcs_costed, nodes_final, 
    #         keep_node_ids=used_nodes,
    #         sample=len(arcs_costed) > 10000,  # 大图自动采样
    #         sample_ratio=0.1
    #     )
        
    #     if sanity_result['has_negative_cycle']:
    #         # 导出负环报告
    #         export_negative_cycle_report(sanity_result, out_dir)
    #         raise ValueError(f"检测到负环！环长度: {len(sanity_result['cycle_path'])}")
    #     else:
    #         print(f"[solver-graph] 负环检测通过 (检查了 {sanity_result['stats']['nodes']} 个节点, {sanity_result['stats']['arcs']} 条弧)")

    # 11) 落盘
    arc_keep_cols = (
        ["arc_id","arc_type","from_node_id","to_node_id","cost","capacity"] +
        [c for c in ["t","i","j","k","tau","tau_total","tau_tochg","tau_chg","de","de_tochg","l","l_to","level","is_last_step","req_key"]
         if c in arcs_costed.columns] +
        [c for c in ["coef_rep","coef_chg_travel","coef_chg_occ","coef_svc_gate","coef_total"]
         if c in arcs_costed.columns]
    )
    arcs_final = arcs_costed[arc_keep_cols].drop_duplicates(subset=["arc_id"]).reset_index(drop=True)

    out_dir_p = Path(out_dir)
    _ensure_dir(out_dir_p)
    nodes_path = out_dir_p / "nodes.parquet"
    arcs_path  = out_dir_p / "arcs.parquet"
    meta_path  = out_dir_p / "meta.json"

    nodes_final.to_parquet(nodes_path, index=False)
    arcs_final.to_parquet(arcs_path, index=False)

    meta = {
        "t0": int(t0),
        "H": int(H),
        "t_hi": int(t_hi),
        "keep_on": keep_on,
        "require_bwd": bool(require_bwd),
        "add_sink": bool(add_sink),
        "cap_infinite": float(cap_infinite),
        "nodes": len(nodes_final),
        "arcs": len(arcs_final),
        "sup_total": float(nodes_with_supply["supply"].clip(lower=0).sum()),
        "sink_node_id": int(sink_id) if sink_id is not None else None,
        "paths": {"nodes": str(nodes_path), "arcs": str(arcs_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[solver-graph] done.")
    print(f"  use cfg: t0={t0} (start_step), H={H} (window_length), t_hi={t_hi}")
    print(f"  nodes: {len(nodes_final):,} -> {nodes_path}")
    print(f"  arcs : {len(arcs_final):,} -> {arcs_path}")
    print(f"  meta : {meta_path}")
    return {"nodes": str(nodes_path), "arcs": str(arcs_path), "meta": str(meta_path)}


# ----------------------------
# CLI：参数全部可选，默认取配置
# ----------------------------
def _parse_bool(x: str) -> bool:
    return str(x).lower() in {"1","true","y","yes","on"}

def main():
    cfg = get_config()
    ap = argparse.ArgumentParser(description="Build graph for independent min-cost-flow / graph optimizer (02~06). Uses config defaults when args omitted.")
    ap.add_argument("--t0", type=int, default=None, help="窗口起点时间步（默认 cfg.time_soc.start_step）")
    ap.add_argument("--H", type=int, default=None, help="窗口长度 H（默认 cfg.time_soc.window_length）")
    ap.add_argument("--keep-on", type=str, default="from", choices=["from","both","either"], help="连通性裁剪保留规则")
    ap.add_argument("--require-bwd", type=_parse_bool, default=False, help="是否启用 BWD 与终端可达约束")
    ap.add_argument("--add-sink", type=_parse_bool, default=True, help="是否添加超级汇点 to_sink 弧以平衡供需")
    ap.add_argument("--cost-to-sink", type=float, default=0.0, help="到汇点弧的成本")
    ap.add_argument("--cap-infinite", type=float, default=1e12, help="缺省弧容量（用于非 gate/occ 弧）")
    ap.add_argument("--check-neg-cycles", type=_parse_bool, default=None, help="是否检测负环（默认 cfg.solver.check_negative_cycles）")
    ap.add_argument("--no-r1-prune", type=_parse_bool, default=False, help="禁用R1重定位弧定向裁剪")
    ap.add_argument("--out", type=str, default="data/solver_graph", help="输出目录")
    args = ap.parse_args()

    # 临时修改配置
    if args.check_neg_cycles is not None:
        cfg.solver.check_negative_cycles = bool(args.check_neg_cycles)
    if args.no_r1_prune:
        cfg.r1_prune.enabled = False
    
    build_solver_graph(
        t0=args.t0,
        H=args.H,
        keep_on=args.keep_on,
        require_bwd=bool(args.require_bwd),
        add_sink=bool(args.add_sink),
        cost_to_sink=float(args.cost_to_sink),
        cap_infinite=float(args.cap_infinite),
        out_dir=args.out,
    )

if __name__ == "__main__":
    main()
