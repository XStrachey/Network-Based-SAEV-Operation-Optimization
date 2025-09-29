# reachability.py  (base_i2k 版)
# 可达性策略：ALL-NEAREST（必须有足够电量到达“最近N个”里最远的那个）
from __future__ import annotations
import json, math, pickle, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from config.network_config import get_network_config as get_config
from utils.energy_utils import get_energy_rate, compute_energy_consumption, compute_multi_timestep_energy_consumption
from utils.grid_utils import GridIndexers, load_indexer, ensure_dir, first_present
from utils.data_loader import load_base_i2k

def build_zone_station_best(cfg, gi: GridIndexers) -> pd.DataFrame:
    """base_i2k 去重：每个 (i,k) 仅保留最短距离条目，并估算 tau_steps（按平均速度）。"""
    i2k = load_base_i2k(cfg); i2k = i2k[i2k["i"].isin(gi.zones)].copy()
    i2k["_dist_sort"] = i2k["dist_km"].fillna(np.inf)
    i2k = i2k.sort_values(["i","k","_dist_sort","k"], kind="stable")
    i2k = i2k.groupby(["i","k"], as_index=False).first().drop(columns=["_dist_sort"])
    dt = int(cfg.time_soc.dt_minutes)
    avg_speed_kmh = float(cfg.basic.avg_speed_kmh)
    minutes = i2k["dist_km"] / max(1e-9, (avg_speed_kmh / 60.0))
    tau_steps = np.ceil(minutes / max(1, dt)).astype("Int64")
    i2k["tau_steps"] = tau_steps.fillna(1).astype(int)
    return i2k[["i","k","dist_km","tau_steps"]]

def select_nearest_stations(cfg, best: pd.DataFrame) -> tuple[Dict[int, List[int]], pd.DataFrame]:
    """
    仅按 dist_km 为每个 i 选前 n 个站。返回 (nearest_map, pruning_stats_df)。
    stats: i, cand_total, kept_n, dropped_n, min/max 距离，threshold_dist, nan_in_cands
    """
    n = cfg.pruning.charge_nearest_station_n or 8

    out: Dict[int, List[int]] = {}
    rows = []
    tmp = best.copy()
    tmp["_dist_sort"] = tmp["dist_km"].fillna(np.inf)
    tmp = tmp.sort_values(["i","_dist_sort","k"], kind="stable")

    for i, grp in tmp.groupby("i", sort=False):
        cand_total = len(grp); nan_in = int(grp["dist_km"].isna().sum())
        kept = grp.head(n)
        kept_ids = kept["k"].astype(int).tolist()
        out[int(i)] = kept_ids
        kept_n = len(kept_ids); dropped_n = max(0, cand_total - kept_n)
        min_all = float(np.nanmin(grp["dist_km"].values)) if cand_total>0 else None
        max_all = float(np.nanmax(grp["dist_km"].values)) if cand_total>0 else None
        min_kept = float(np.nanmin(kept["dist_km"].values)) if kept_n>0 else None
        max_kept = float(np.nanmax(kept["dist_km"].values)) if kept_n>0 else None
        rows.append({
            "i": int(i), "cand_total": int(cand_total), "kept_n": int(kept_n), "dropped_n": int(dropped_n),
            "min_dist_all": min_all, "max_dist_all": max_all,
            "min_dist_kept": min_kept, "max_dist_kept": max_kept,
            "threshold_dist": max_kept, "nan_in_cands": nan_in
        })

    stats_df = pd.DataFrame(rows, columns=[
        "i","cand_total","kept_n","dropped_n","min_dist_all","max_dist_all",
        "min_dist_kept","max_dist_kept","threshold_dist","nan_in_cands"
    ])
    return out, stats_df

# —— 能耗基表（用于核对） —— #
def build_zone_station_energy_table(cfg, gi: GridIndexers, best: pd.DataFrame, pairs_map: Dict[int, List[int]]) -> pd.DataFrame:
    """返回列：t, i, k, dist_km, tau_steps, de_tochg（最近集合内每个站的到站能耗）"""
    bx = best.set_index(["i","k"])
    soc_levels = np.array(gi.socs, dtype=int)
    soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100
    rows = []
    for t in gi.times:
        for i, ks in pairs_map.items():
            for k in ks:
                if (i, k) not in bx.index: continue
                row = bx.loc[(i, k)]
                d = row["dist_km"]; tau_steps = int(row["tau_steps"]) if not pd.isna(row["tau_steps"]) else 1
                de = compute_multi_timestep_energy_consumption(cfg, float(d) if not pd.isna(d) else float("nan"), int(t), tau_steps, soc_step, "de_per_km_tochg")
                rows.append({
                    "t": int(t), "i": int(i), "k": int(k),
                    "dist_km": (float(d) if not pd.isna(d) else np.nan),
                    "tau_steps": int(tau_steps),
                    "de_tochg": (int(de) if not (isinstance(de,float) and math.isnan(de)) else np.nan)
                })
    return pd.DataFrame(rows, columns=["t","i","k","dist_km","tau_steps","de_tochg"])

def build_energy_agg(energy_kept: pd.DataFrame) -> pd.DataFrame:
    """聚合出每个 (t,i) 的 min/max 能耗及对应站点。"""
    if energy_kept.empty:
        return pd.DataFrame(columns=["t","i","n_nearest","min_de_any","k_argmin","max_de_all","k_argmax"])
    tmp = energy_kept.dropna(subset=["de_tochg"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["t","i","n_nearest","min_de_any","k_argmin","max_de_all","k_argmax"])
    n_nearest = energy_kept.groupby(["t","i"])["k"].count().rename("n_nearest")
    idx_min = tmp.groupby(["t","i"])["de_tochg"].idxmin()
    idx_max = tmp.groupby(["t","i"])["de_tochg"].idxmax()
    min_df = tmp.loc[idx_min, ["t","i","k","de_tochg"]].rename(columns={"k":"k_argmin","de_tochg":"min_de_any"})
    max_df = tmp.loc[idx_max, ["t","i","k","de_tochg"]].rename(columns={"k":"k_argmax","de_tochg":"max_de_all"})
    out = min_df.merge(max_df, on=["t","i"], how="outer").merge(n_nearest, on=["t","i"], how="outer")
    return out.sort_values(["t","i"]).reset_index(drop=True)

# —— 可达性表（ALL-NEAREST 策略） —— #
def build_reachability_table(cfg, gi: GridIndexers, best: pd.DataFrame, nearest_map: Dict[int,List[int]]) -> tuple[pd.DataFrame, List[int]]:
    """
    对每个 (t, zone=j)：
      计算候选 ks = nearest_map[j] 的 de_tochg 列表；
      令 req_de_all = max(de_tochg)；若某些 de 为 NaN，则视为该站不可计算 → 不计入 n_finite；
      仅当 n_finite == n_expected 且 soc >= req_de_all 时，reachable = 1（能到全部最近站）。
    同时输出 num_reachable = sum(de_tochg <= soc) 供诊断。
    """
    bx = best.set_index(["i","k"])
    rows = []
    zones_no_station = []
    soc_levels = np.array(gi.socs, dtype=int)
    soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100

    for t in gi.times:
        for j in gi.zones:
            cand_k = nearest_map.get(j, [])
            if not cand_k:
                zones_no_station.append(j)
                for l in gi.socs:
                    rows.append((t, j, int(l), 0, 0, None, None, 0, 0))
                continue

            de_vals = []
            for k in cand_k:
                if (j, k) not in bx.index:
                    de_vals.append(np.nan); continue
                row = bx.loc[(j, k)]
                d = float(row["dist_km"]) if not pd.isna(row["dist_km"]) else float("nan")
                tau_steps = int(row["tau_steps"]) if not pd.isna(row["tau_steps"]) else 1
                de = compute_multi_timestep_energy_consumption(cfg, d, int(t), tau_steps, soc_step, "de_per_km_tochg")
                de_vals.append(float(de) if not (isinstance(de,float) and math.isnan(de)) else np.nan)

            de_arr = np.array(de_vals, dtype=float)
            n_expected = len(cand_k)
            finite_mask = ~np.isnan(de_arr)
            n_finite = int(finite_mask.sum())

            req_de_all = float(np.max(de_arr[finite_mask])) if n_finite>0 else None
            min_de_any = float(np.min(de_arr[finite_mask])) if n_finite>0 else None

            for l in gi.socs:
                l_int = int(l)
                cnt = int(np.sum(de_arr[finite_mask] <= l_int)) if n_finite>0 else 0
                reachable_all = int( (n_finite == n_expected) and (req_de_all is not None) and (l_int >= req_de_all + 10) ) # 需要一点冗余，否则会出现0%的节点
                rows.append((
                    int(t), int(j), l_int,
                    reachable_all,                # reachable（新语义：能到所有最近站）
                    cnt,                          # num_reachable（诊断：能到几个）
                    req_de_all,                   # 需要达到全部最近站的能耗阈值（max）
                    min_de_any,                   # 达到任一最近站的最小能耗（min）
                    n_expected, n_finite          # 期望站点数 / 有效能耗数
                ))

    cols = ["t","zone","soc","reachable","num_reachable","req_de_all","min_de_any","n_expected","n_finite"]
    result_df = pd.DataFrame(rows, columns=cols)
    return result_df, zones_no_station

# —— 主流程 —— #
def main():
    cfg = get_config(); gi = load_indexer()
    out_dir = Path("data/intermediate"); ensure_dir(out_dir)

    print("[reachability] Using base_i2k …")
    best = build_zone_station_best(cfg, gi)
    best_path = out_dir/"zone_station_best.parquet"
    best.to_parquet(best_path, index=False)
    best.to_csv(out_dir/"zone_station_best.csv", index=False)

    nearest_map, prune_stats = select_nearest_stations(cfg, best)
    (out_dir/"nearest_stations.json").write_text(json.dumps({str(k):v for k,v in nearest_map.items()}, indent=2))
    prune_stats_path = out_dir/"nearest_stations_pruning.csv"
    prune_stats.to_csv(prune_stats_path, index=False)

    # 能耗基表（保留站集合）
    energy_kept = build_zone_station_energy_table(cfg, gi, best, nearest_map)
    energy_kept_path = out_dir/"zone_station_energy_kept.csv"
    energy_kept.to_csv(energy_kept_path, index=False)

    # 聚合（每 t,zone 的 min/max 及对应站点）
    energy_agg = build_energy_agg(energy_kept)
    energy_agg_path = out_dir/"zone_energy_nearest_agg.csv"
    energy_agg.to_csv(energy_agg_path, index=False)

    # 可达性（ALL-NEAREST）
    reach, zones_no_station = build_reachability_table(cfg, gi, best, nearest_map)
    reach_path = out_dir/"reachability.parquet"
    reach.to_parquet(reach_path, index=False)
    reach.to_csv(out_dir/"reachability.csv", index=False)

    # 汇总统计
    num_grid_nodes = gi.n_z * gi.n_t * gi.n_l
    num_reachable_nodes = int((reach["reachable"] == 1).sum())
    nearest_n = int(cfg.pruning.charge_nearest_station_n) or 8
    cand_total_all = int(prune_stats["cand_total"].sum()) if not prune_stats.empty else 0
    kept_total_all = int(prune_stats["kept_n"].sum()) if not prune_stats.empty else 0
    dropped_total_all = int(prune_stats["dropped_n"].sum()) if not prune_stats.empty else 0
    zones_with_candidates = int((prune_stats["cand_total"] > 0).sum()) if not prune_stats.empty else 0
    zones_pruned = int((prune_stats["dropped_n"] > 0).sum()) if not prune_stats.empty else 0
    nan_dist_pairs = int(prune_stats["nan_in_cands"].sum()) if not prune_stats.empty else 0

    meta = {
        "zones": gi.n_z,
        "soc_levels": gi.n_l,
        "rows_reachability": len(reach),
        "num_grid_nodes": num_grid_nodes,
        "num_reachable_nodes": num_reachable_nodes,
        "prune_ratio": 1.0 - (num_reachable_nodes / num_grid_nodes) if num_grid_nodes > 0 else None,
        # 最近站裁剪
        "nearest_k": nearest_n,
        "nearest_metric": "dist_km",
        "nearest_candidates_total": cand_total_all,
        "nearest_kept_total": kept_total_all,
        "nearest_dropped_total": dropped_total_all,
        "zones_with_candidates": zones_with_candidates,
        "zones_pruned": zones_pruned,
        "nan_dist_pairs": nan_dist_pairs,
        # 新的可达性语义
        "reachability_policy": "all_nearest",
        "reachability_threshold": "max_de_tochg_among_nearest",
        # 路径
        "paths": {
            "zone_station_best": str(best_path),
            "nearest_stations": str(out_dir / "nearest_stations.json"),
            "nearest_stations_pruning": str(prune_stats_path),
            "reachability": str(reach_path),
            "zone_station_energy_kept": str(energy_kept_path),
            "zone_energy_nearest_agg": str(energy_agg_path),
        },
        "source": "base_i2k"
    }
    (out_dir/"reachability_meta.json").write_text(json.dumps(meta, indent=2))

    print(
        "[reachability] Done (ALL-NEAREST). "
        f"rows: {len(reach)} | grid nodes: {num_grid_nodes} | reachable(all): {num_reachable_nodes} "
        f"| prune ratio: {meta['prune_ratio']:.3f} | nearest_k: {nearest_n} "
        f"| cand_total: {cand_total_all} | kept: {kept_total_all} | dropped: {dropped_total_all} "
        f"| zones(no station): {len(zones_no_station)} | zones(pruned): {zones_pruned}"
    )
    print(f"[reachability] energy basis: {out_dir/'zone_station_energy_kept.csv'} "
          f"& agg: {out_dir/'zone_energy_nearest_agg.csv'}")

if __name__ == "__main__":
    main()
