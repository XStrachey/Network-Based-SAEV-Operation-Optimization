# solve_graph_mincost.py
# 独立最小费用流求解器（零参数可跑）
# 默认从 data/solver_graph/meta.json 读取图文件；找不到就尝试 data/solver_graph/nodes.* / arcs.*
# 也可以手动指定：--nodes --arcs；结果写到 outputs/

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# 依赖：pulp（GLPK/CBC）
try:
    import pulp
except ImportError as e:
    raise SystemExit("缺少依赖 pulp，请先安装：pip install pulp") from e


# -----------------------------
# 路径自动发现
# -----------------------------
def _infer_graph_paths(
    nodes: Optional[str],
    arcs: Optional[str],
    meta: Optional[str],
    default_dir: str = "data/solver_graph",
) -> Tuple[str, str]:
    """
    优先级：
      1) 显式 --nodes/--arcs
      2) meta.json -> paths.nodes/paths.arcs
      3) default_dir/nodes.(parquet|csv) 和 arcs.(parquet|csv)
    """
    if nodes and arcs:
        return nodes, arcs

    # 2) meta.json
    meta_candidates = []
    if meta:
        meta_candidates.append(Path(meta))
    meta_candidates.append(Path(default_dir) / "meta.json")

    for mp in meta_candidates:
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                p = m.get("paths", {})
                n = p.get("nodes")
                a = p.get("arcs")
                if n and a and Path(n).exists() and Path(a).exists():
                    return str(n), str(a)
            except Exception:
                pass  # 继续兜底

    # 3) 直接找默认文件
    dd = Path(default_dir)
    if dd.exists():
        nodes_guess = None
        arcs_guess = None
        for ext in (".parquet", ".pq", ".csv"):
            if (dd / f"nodes{ext}").exists():
                nodes_guess = dd / f"nodes{ext}"
                break
        for ext in (".parquet", ".pq", ".csv"):
            if (dd / f"arcs{ext}").exists():
                arcs_guess = dd / f"arcs{ext}"
                break
        if nodes_guess and arcs_guess:
            return str(nodes_guess), str(arcs_guess)

    raise FileNotFoundError(
        "无法自动定位图文件。请确保已运行 build_solver_graph.py，"
        "或者使用 --nodes 与 --arcs 显式指定文件路径。"
    )


# -----------------------------
# I/O 与预处理
# -----------------------------
def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except Exception as e:
            raise SystemExit(
                f"读取 {p} 失败。若未安装 parquet 引擎，请先安装：pip install pyarrow 或 fastparquet。"
            ) from e
    return pd.read_csv(p)


def _ensure_arc_id(arcs: pd.DataFrame) -> pd.DataFrame:
    df = arcs.copy()
    if "arc_id" in df.columns and df["arc_id"].notna().all():
        return df
    key_cols = [c for c in ["arc_type", "from_node_id", "to_node_id"] if c in df.columns]
    if len(key_cols) < 2:
        df["arc_id"] = (df.index.values.astype("int64") + 1).astype("int64")
        return df
    key = df[key_cols].astype(str).agg("|".join, axis=1)
    h = pd.util.hash_pandas_object(key, index=False).astype("int64")
    df["arc_id"] = (h & np.int64(0x7FFFFFFFFFFFFFFF)).astype("int64")
    return df


def _sanitize_graph(nodes: pd.DataFrame, arcs: pd.DataFrame, big_cap: float = 1e12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = nodes.copy()
    a = arcs.copy()

    for col in ["node_id", "supply"]:
        if col not in n.columns:
            raise ValueError(f"[nodes] 缺少必需列: {col}")
    for col in ["from_node_id", "to_node_id", "cost", "capacity"]:
        if col not in a.columns:
            raise ValueError(f"[arcs] 缺少必需列: {col}")

    n["node_id"] = n["node_id"].astype("int64")
    n["supply"] = pd.to_numeric(n["supply"], errors="coerce").fillna(0.0).astype(float)

    a = _ensure_arc_id(a)
    a["from_node_id"] = a["from_node_id"].astype("int64")
    a["to_node_id"] = a["to_node_id"].astype("int64")
    a["cost"] = pd.to_numeric(a["cost"], errors="coerce").fillna(0.0).astype(float)
    a["capacity"] = pd.to_numeric(a["capacity"], errors="coerce").fillna(big_cap).clip(lower=0.0)
    # 将无穷大值替换为有限的大值
    a.loc[np.isinf(a["capacity"]), "capacity"] = big_cap
    a["capacity"] = a["capacity"].astype(float)

    used_nodes = set(a["from_node_id"].tolist()) | set(a["to_node_id"].tolist())
    has_supply = set(n.loc[n["supply"].abs() > 0, "node_id"].tolist())
    keep_nodes = used_nodes | has_supply
    n = n[n["node_id"].isin(keep_nodes)].copy()

    splus = float(n["supply"].clip(lower=0).sum())
    sneg = float(-n["supply"].clip(upper=0).sum())
    if abs(splus - sneg) > 1e-6:
        raise ValueError(
            f"[nodes] 供需不平衡：正供给 {splus}, 负需求 {sneg}。"
            "请在建图时启用超级汇点或自行平衡 supply。"
        )
    return n, a


# -----------------------------
# 求解（LP 连续最小费用流）
# -----------------------------
def _pick_solver(name: str = "auto", time_limit: int = 3600, msg: bool = True):
    name = (name or "auto").lower()
    if name in {"glpk", "auto"}:
        try:
            return pulp.GLPK_CMD(msg=msg, options=["--tmlim", str(int(time_limit))])
        except Exception:
            if name == "glpk":
                raise
    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=int(time_limit))


def solve_min_cost_flow(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    *,
    solver: str = "auto",
    time_limit: int = 3600,
    tolerance_zero: float = 1e-9,
    msg: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n, a = _sanitize_graph(nodes, arcs)

    node_ids = n["node_id"].astype("int64").tolist()
    supplies = dict(zip(n["node_id"].astype("int64"), n["supply"].astype(float)))

    a = a.reset_index(drop=True)
    a["__idx__"] = np.arange(len(a), dtype=np.int64)
    idx = a["__idx__"].to_numpy()
    fr = a["from_node_id"].to_numpy(dtype=np.int64)
    to = a["to_node_id"].to_numpy(dtype=np.int64)
    cap = a["capacity"].to_numpy(dtype=float)
    cst = a["cost"].to_numpy(dtype=float)

    prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)
    f = {int(i): pulp.LpVariable(f"f_{int(i)}", lowBound=0.0, upBound=float(cap[k]), cat="Continuous")
         for k, i in enumerate(idx)}

    prob += pulp.lpSum(float(cst[k]) * f[int(i)] for k, i in enumerate(idx)), "total_cost"

    out_map: Dict[int, list] = {int(v): [] for v in node_ids}
    in_map: Dict[int, list] = {int(v): [] for v in node_ids}
    for k, i in enumerate(idx):
        u = int(fr[k]); v = int(to[k])
        if u in out_map: out_map[u].append(int(i))
        if v in in_map:  in_map[v].append(int(i))

    for nid in node_ids:
        prob += (pulp.lpSum(f[i] for i in out_map.get(int(nid), ())) -
                 pulp.lpSum(f[i] for i in in_map.get(int(nid), ())) ==
                 float(supplies.get(int(nid), 0.0))), f"flow_balance_{int(nid)}"

    solver_obj = _pick_solver(solver, time_limit=time_limit, msg=msg)
    status_code = prob.solve(solver_obj)
    status_name = pulp.LpStatus.get(status_code, str(status_code))

    obj = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else np.nan
    flows = np.array([pulp.value(f[int(i)]) for i in idx], dtype=float)
    flows[np.isnan(flows)] = 0.0

    out = a.drop(columns=["__idx__"]).copy()
    out["flow"] = flows
    out.loc[out["flow"].abs() < float(tolerance_zero), "flow"] = 0.0

        # 统计
    total_cost = float((out["flow"] * out["cost"]).sum())
    total_flow = float(out["flow"].sum())

    by_type = {}
    if "arc_type" in out.columns:
        tmp = out.copy()
        tmp["_cost"] = tmp["flow"] * tmp["cost"]
        g = tmp.groupby("arc_type", dropna=False).agg(
            flow_sum=("flow", "sum"),
            cost_sum=("_cost", "sum"),
            arcs=("arc_id", "count"),
        )
        # 用 orient="index" 转 dict，再遍历键值对
        by_type = {}
        for k, v in g.to_dict(orient="index").items():
            key = "nan" if pd.isna(k) else str(k)
            by_type[key] = {
                "flow": float(v["flow_sum"]),
                "cost": float(v["cost_sum"]),
                "arcs": int(v["arcs"]),
            }

    summary = {
        "status": status_name,
        "objective": obj,
        "total_cost": total_cost,
        "total_flow": total_flow,
        "nodes": int(len(n)),
        "arcs": int(len(a)),
        "by_type": by_type,
    }
    return out, summary


# -----------------------------
# CLI（参数可全省略）
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="独立最小费用流求解器（默认读取 data/solver_graph/*）")
    ap.add_argument("--nodes", type=str, default=None, help="nodes 文件（parquet/csv），默认自动发现")
    ap.add_argument("--arcs", type=str, default=None, help="arcs 文件（parquet/csv），默认自动发现")
    ap.add_argument("--meta", type=str, default=None, help="meta.json 路径，默认 data/solver_graph/meta.json")
    ap.add_argument("--out", type=str, default="outputs", help="输出目录")
    ap.add_argument("--solver", type=str, default="auto", choices=["auto", "glpk", "cbc"], help="求解器选择")
    ap.add_argument("--time-limit", type=int, default=3600, help="时间限制（秒）")
    ap.add_argument("--msg", type=int, default=1, help="求解日志（1/0）")
    ap.add_argument("--zero-tol", type=float, default=1e-9, help="将小于该阈值的流量置零")
    args = ap.parse_args()

    nodes_path, arcs_path = _infer_graph_paths(args.nodes, args.arcs, args.meta)
    nodes = _read_table(nodes_path)
    arcs = _read_table(arcs_path)

    flows_df, summary = solve_min_cost_flow(
        nodes, arcs,
        solver=args.solver,
        time_limit=args.time_limit,
        tolerance_zero=float(args.zero_tol),
        msg=bool(args.msg),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    flows_pq = out_dir / "flows.parquet"
    flows_csv = out_dir / "flows.csv"
    meta_path = out_dir / "solve_summary.json"

    try:
        flows_df.to_parquet(flows_pq, index=False)
        flows_written = str(flows_pq)
    except Exception:
        flows_df.to_csv(flows_csv, index=False)
        flows_written = str(flows_csv)

    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[solver] done.")
    print(f"  status      = {summary['status']}")
    print(f"  objective   = {summary['objective']:.6g}")
    print(f"  total_cost  = {summary['total_cost']:.6g}")
    print(f"  total_flow  = {summary['total_flow']:.6g}")
    print(f"  arcs output -> {flows_written}")
    print(f"  summary     -> {meta_path}")


if __name__ == "__main__":
    main()
