# _02_build_time_soc_grid.py
# 仅使用 config 中给定的 zone 集合作为基础输入，构建 (zone, time, SOC) 网格与索引器
# 产物保存到 data/intermediate/
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _01_network_config import get_network_config as get_config
# 与 04/05 保持一致：统一使用 grid_utils 内的公共实现，避免重复类导致的 pickle 反序列化不兼容
from grid_utils import GridIndexers, ensure_dir, first_present


# ----------------------------
# 基础加载
# ----------------------------
def load_zones_from_config(cfg) -> List[int]:
    """从 config.paths.zones 读取 zone 集合；不再从 OD / stations 推断"""
    zpath = Path(cfg.paths.zones)
    if not zpath.exists():
        raise FileNotFoundError(f"Zones file not found: {zpath}")

    df = pd.read_csv(zpath)
    col_zone = first_present(df.columns, ["zone", "taz", "zone_id"])
    if col_zone is None:
        raise ValueError("zones.csv 必须包含列 'zone'（或同义列 taz/zone_id）。")

    zones = df[col_zone].dropna().astype(int).unique().tolist()
    zones = sorted(zones)
    if len(zones) == 0:
        raise ValueError("zones.csv 解析为空。")
    return zones


# ----------------------------
# 网格索引器
# ----------------------------
def build_indexers(cfg) -> GridIndexers:
    """
    构建三维索引器（zone 列表直接来自 zones.csv）
    时间维度：从 start_step 到 end_step + overhang_steps（含端点）
    """
    zones = load_zones_from_config(cfg)

    t_start = int(cfg.time_soc.start_step)
    t_end   = int(cfg.time_soc.end_step)
    over    = int(cfg.time_soc.overhang_steps)
    ts = list(range(t_start, t_end + over + 1))  # 注意：+1 以包含末端

    socs = [int(x) for x in cfg.time_soc.soc_levels]

    zone_to_idx: Dict[int, int] = {z: k for k, z in enumerate(zones)}
    time_to_idx: Dict[int, int] = {t: k for k, t in enumerate(ts)}
    soc_to_idx:  Dict[int, int] = {l: k for k, l in enumerate(socs)}

    gi = GridIndexers(
        zones=zones,
        times=ts,
        socs=socs,
        zone_to_idx=zone_to_idx,
        time_to_idx=time_to_idx,
        soc_to_idx=soc_to_idx,
        n_z=len(zones),
        n_t=len(ts),
        n_l=len(socs),
    )
    return gi


def materialize_nodes_dataframe(gi: GridIndexers) -> pd.DataFrame:
    """
    生成 nodes 表（node_id, zone, t, soc）
    node_id 编码需与 GridIndexers.id_of 保持一致：
      id = (zi * n_t + ti) * n_l + li
    """
    z = np.repeat(np.array(gi.zones, dtype=np.int64), gi.n_t * gi.n_l)
    t = np.tile(np.repeat(np.array(gi.times, dtype=np.int64), gi.n_l), gi.n_z)
    l = np.tile(np.array(gi.socs,  dtype=np.int64), gi.n_z * gi.n_t)

    # 与 id_of 一致的顺序编码
    zi = np.repeat(np.arange(gi.n_z, dtype=np.int64), gi.n_t * gi.n_l)
    ti = np.tile(np.repeat(np.arange(gi.n_t, dtype=np.int64), gi.n_l), gi.n_z)
    li = np.tile(np.arange(gi.n_l, dtype=np.int64), gi.n_z * gi.n_t)
    node_id = (zi * gi.n_t + ti) * gi.n_l + li

    return pd.DataFrame(
        {"node_id": node_id, "zone": z, "t": t, "soc": l},
        columns=["node_id", "zone", "t", "soc"]
    )


# ----------------------------
# 初始库存
# ----------------------------
def load_initial_inventory(cfg, gi: GridIndexers) -> pd.DataFrame:
    """
    读取初始车量库存 cfg.paths.fleet_init -> DataFrame:
      columns: zone, soc, count  （仅 t = start_step）
    自动校验与对齐 zone/soc，缺失组合填 0；重复 (zone,soc) 聚合；负数计数截为 0。
    """
    fp = Path(cfg.paths.fleet_init)
    if not fp.exists():
        raise FileNotFoundError(f"Fleet initial inventory not found: {fp}（请提供 data/fleet_init.csv）")

    df = pd.read_csv(fp)
    col_zone = first_present(df.columns, ["zone", "taz", "zone_id"])
    col_soc  = first_present(df.columns, ["soc", "soc_level"])
    col_cnt  = first_present(df.columns, ["count", "quantity"])
    if col_zone is None or col_soc is None or col_cnt is None:
        raise ValueError("fleet_init.csv 必须包含列: zone(或 taz), soc(或 soc_level), count(或 quantity)。")

    df = df[[col_zone, col_soc, col_cnt]].rename(
        columns={col_zone: "zone", col_soc: "soc", col_cnt: "count"}
    )
    df["zone"]  = df["zone"].astype(int)
    df["soc"]   = df["soc"].astype(int)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0).astype(float)
    df.loc[df["count"] < 0, "count"] = 0.0
    df = df.groupby(["zone", "soc"], as_index=False)["count"].sum()

    # 校验 zone/soc
    invalid_z = set(df["zone"].unique()) - set(gi.zones)
    if invalid_z:
        raise ValueError(f"fleet_init 包含未知 zones（不在 zones.csv 中）: {sorted(invalid_z)[:10]} ...")
    invalid_soc = set(df["soc"].unique()) - set(gi.socs)
    if invalid_soc:
        raise ValueError(f"fleet_init 包含未知 SOC 水平（不在 config.time_soc.soc_levels 中）: {sorted(invalid_soc)}")

    # 补全所有 (zone, soc) 组合，并赋 t = start_step
    full = (
        pd.MultiIndex.from_product([gi.zones, gi.socs], names=["zone", "soc"])
        .to_frame(index=False)
        .merge(df, on=["zone", "soc"], how="left")
        .fillna({"count": 0.0})
    )
    full["t"] = gi.times[0]
    return full[["zone", "soc", "t", "count"]]


# ----------------------------
# 落盘
# ----------------------------
def save_artifacts(cfg, gi: GridIndexers, nodes: pd.DataFrame, V0: pd.DataFrame) -> None:
    inter_dir = Path("data/intermediate")
    ensure_dir(inter_dir)

    # nodes
    nodes_path = inter_dir / "nodes.parquet"
    nodes.to_parquet(nodes_path, index=False)

    # 索引映射（便于调试/外部使用）
    (inter_dir / "zone_index.json").write_text(json.dumps(gi.zone_to_idx, indent=2))
    (inter_dir / "soc_index.json").write_text(json.dumps(gi.soc_to_idx, indent=2))
    (inter_dir / "time_index.json").write_text(json.dumps(gi.time_to_idx, indent=2))

    # 索引器（pickle，用于 grid_utils.load_indexer 读取）
    with open(inter_dir / "node_indexer.pkl", "wb") as f:
        pickle.dump(gi, f)

    # 初始库存
    inv_path = inter_dir / "initial_inventory.parquet"
    V0.to_parquet(inv_path, index=False)

    log = {
        "n_zones": gi.n_z,
        "n_times": gi.n_t,
        "n_socs": gi.n_l,
        "num_nodes": gi.num_nodes,
        "nodes_path": str(nodes_path),
        "initial_inventory_path": str(inv_path),
        "init_total": float(V0["count"].sum()),
        "init_soc_levels_used": sorted(V0.loc[V0["count"] > 0, "soc"].unique().tolist()),
        "time_range": [int(min(gi.times)), int(max(gi.times))],
        "overhang_steps": int(get_config().time_soc.overhang_steps),
    }
    (inter_dir / "grid_summary.json").write_text(json.dumps(log, indent=2))

    if get_config().solver.verbose:
        print("[build_time_soc_grid] Saved artifacts:")
        for k, v in log.items():
            print(f"  - {k}: {v}")


# ----------------------------
# CLI
# ----------------------------
def main():
    cfg = get_config()
    if cfg.solver.verbose:
        print("[build_time_soc_grid] Building indexers from config.paths.zones ...")

    gi = build_indexers(cfg)
    if cfg.solver.verbose:
        print(f"[build_time_soc_grid] Zones={gi.n_z}, Times={gi.n_t}, SOC levels={gi.n_l}, Nodes={gi.num_nodes}")

    nodes = materialize_nodes_dataframe(gi)
    if cfg.solver.verbose:
        print("[build_time_soc_grid] Loading initial inventory ...")

    V0 = load_initial_inventory(cfg, gi)
    if cfg.solver.verbose:
        print(f"[build_time_soc_grid] Initial vehicles total: {V0['count'].sum():.0f}")

    save_artifacts(cfg, gi, nodes, V0)


if __name__ == "__main__":
    main()
