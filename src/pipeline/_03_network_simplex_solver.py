# _10_network_simplex_solver.py
# 使用 Successive Shortest Path（带势的最短路增广）求解最小费用流
# 读取 _08_build_solver_graph.py 生成的图数据，输出 flows.parquet / flows.csv 与 solve_summary.json

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from heapq import heappush, heappop

# （保持与原脚本一致的导入；即便未使用，也不影响）
from config.network_config import get_network_config  # noqa: F401


# ========== 工具与校验 ==========

REQUIRED_NODE_COLS = {"node_id", "supply"}
REQUIRED_ARC_COLS = {"from_node_id", "to_node_id", "capacity", "cost"}


def _validate_schema(nodes: pd.DataFrame, arcs: pd.DataFrame):
    missing_nodes = REQUIRED_NODE_COLS - set(nodes.columns)
    missing_arcs = REQUIRED_ARC_COLS - set(arcs.columns)
    if missing_nodes:
        raise ValueError(f"nodes 缺少必要列: {sorted(missing_nodes)}")
    if missing_arcs:
        raise ValueError(f"arcs 缺少必要列: {sorted(missing_arcs)}")

    # 基本类型与取值检查
    for c in ["capacity", "cost"]:
        if not np.issubdtype(arcs[c].dtype, np.number):
            raise ValueError(f"arcs.{c} 必须为数值列")
    if (arcs["capacity"] < 0).any():
        raise ValueError("arcs.capacity 不能为负")

    if not np.issubdtype(nodes["supply"].dtype, np.number):
        raise ValueError("nodes.supply 必须为数值列")


# ========== 残量图边结构 ==========

@dataclass
class Edge:
    to: int
    rev: int          # 对端邻接表中的“反向边索引”
    cap: float
    cost: float
    arc_idx: int      # 原始弧索引（用于回写 flow）；反向边同样保存这个索引
    is_forward: bool  # True 表示“原始正向边”的残量，False 为其反向残量边


class NetworkSimplexSolver:
    """
    使用 Successive Shortest Path (SSP) + 潜势（势）维护约化费用：
      - 保证每次 Dijkstra 在非负权上运行；
      - 每次从某个尚有供给的 s 出发，找到到某个尚有需求的 t 的最短路并增广；
      - 直到所有供给分配完（或触发时限/不可达 => INFEASIBLE/TIME_LIMIT）。
    """

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, *, msg: bool = False):
        self.nodes = nodes.copy()
        self.arcs = arcs.copy()
        _validate_schema(self.nodes, self.arcs)

        # 基本规模
        self.n_nodes = len(self.nodes)
        self.n_arcs = len(self.arcs)
        self.msg = msg

        # 节点映射：node_id -> [0..n-1]
        self.node_ids = self.nodes["node_id"].tolist()
        self.id2idx = {nid: i for i, nid in enumerate(self.node_ids)}

        # 将 supply 按索引对齐为向量 b（>0 供给；<0 需求）
        self.b = np.zeros(self.n_nodes, dtype=float)
        for i, nid in enumerate(self.node_ids):
            self.b[i] = float(self.nodes.loc[self.nodes["node_id"] == nid, "supply"].iloc[0])

        # 初始流（按原始弧顺序回写）
        self.flow = np.zeros(self.n_arcs, dtype=float)

        # 残量图（邻接表）
        self.G: List[List[Edge]] = [list() for _ in range(self.n_nodes)]
        # 为每条原始弧记录其“正向残量边”的位置：arc_idx -> (u_idx, edge_pos)
        self.fwd_pos: List[Tuple[int, int]] = [(-1, -1)] * self.n_arcs

        # 构建残量图
        self._build_residual()

    # ---- 残量网络 ----
    def _add_edge(self, u: int, v: int, cap: float, cost: float, arc_idx: int):
        """为原始弧 u->v 添加一对残量边"""
        fwd_index = len(self.G[u])
        rev_index = len(self.G[v])
        self.G[u].append(Edge(to=v, rev=rev_index, cap=cap, cost=cost, arc_idx=arc_idx, is_forward=True))
        self.G[v].append(Edge(to=u, rev=fwd_index, cap=0.0, cost=-cost, arc_idx=arc_idx, is_forward=False))
        # 记录原始正向边位置（用于最终提取 flow）
        self.fwd_pos[arc_idx] = (u, fwd_index)

    def _build_residual(self):
        for i, a in self.arcs.reset_index(drop=True).iterrows():
            u_id = a["from_node_id"]
            v_id = a["to_node_id"]
            if u_id not in self.id2idx or v_id not in self.id2idx:
                raise ValueError(f"弧 {i} 的端点不在 nodes 中: {u_id}->{v_id}")
            u = self.id2idx[u_id]
            v = self.id2idx[v_id]
            cap = float(a["capacity"])
            cost = float(a["cost"])
            self._add_edge(u, v, cap, cost, arc_idx=i)

    # ---- 诊断 ----
    def _max_node_imbalance(self) -> float:
        """基于 self.flow 计算 max |(出-入)-supply|"""
        df = pd.DataFrame({
            "from": self.arcs["from_node_id"].values,
            "to": self.arcs["to_node_id"].values,
            "f": self.flow
        })
        out_ = df.groupby("from")["f"].sum()
        in_ = df.groupby("to")["f"].sum()
        net = out_.sub(in_, fill_value=0.0)
        supply = self.nodes.set_index("node_id")["supply"]
        net = net.reindex(supply.index, fill_value=0.0)
        return float((net - supply).abs().max())

    # ---- Dijkstra（约化费用） ----
    def _dijkstra(self, s: int, pi: np.ndarray) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
        """
        在 cap>0 的边上用约化费用 c' = c + pi[u] - pi[v] 跑 Dijkstra。
        返回 dist 与 prev（prev[v] = (u, edge_idx)）。
        """
        n = self.n_nodes
        INF = 1e100
        dist = np.full(n, INF, dtype=float)
        prev: List[Optional[Tuple[int, int]]] = [None] * n
        dist[s] = 0.0
        hq: List[Tuple[float, int]] = [(0.0, s)]

        while hq:
            d, u = heappop(hq)
            if d != dist[u]:
                continue
            for ei, e in enumerate(self.G[u]):
                if e.cap <= 1e-12:
                    continue
                rc = e.cost + pi[u] - pi[e.to]  # reduced cost
                nd = d + rc
                if nd + 1e-15 < dist[e.to]:
                    dist[e.to] = nd
                    prev[e.to] = (u, ei)
                    heappush(hq, (nd, e.to))
        return dist, prev

    def _extract_path(self, prev: List[Optional[Tuple[int, int]]], t: int) -> List[Tuple[int, int]]:
        """回溯 prev 得到从 s 到 t 的 (u, edge_idx) 路径列表"""
        path: List[Tuple[int, int]] = []
        cur = t
        while prev[cur] is not None:
            u, ei = prev[cur]
            path.append((u, ei))
            cur = u
        path.reverse()
        return path

    def _path_residual_cap(self, path: List[Tuple[int, int]]) -> float:
        r = float("inf")
        for u, ei in path:
            e = self.G[u][ei]
            r = min(r, e.cap)
        return r if np.isfinite(r) else 0.0

    def _augment(self, path: List[Tuple[int, int]], delta: float):
        """沿路径增广：正向残量减 delta，反向残量加 delta"""
        for u, ei in path:
            e = self.G[u][ei]
            v = e.to
            rev = e.rev
            # 更新正向
            e.cap -= delta
            # 更新反向
            rev_edge = self.G[v][rev]
            rev_edge.cap += delta

    def _writeback_flow(self):
        """从残量网络回写 self.flow（反向残量边的容量即当前流量）"""
        for i in range(self.n_arcs):
            u, ei = self.fwd_pos[i]
            if u < 0:
                self.flow[i] = 0.0
                continue
            e = self.G[u][ei]            # 正向残量
            v = e.to
            rev_edge = self.G[v][e.rev]  # 反向残量
            self.flow[i] = float(rev_edge.cap)  # 当前流量

    # ---- 主求解 ----
    def solve(self, *, time_limit: Optional[float] = None) -> Dict[str, Any]:
        start_time = time.time()

        # 供需平衡检查
        total_supply = float(np.clip(self.b, 0, None).sum())
        total_demand = float(-np.clip(self.b, None, 0).sum())
        if abs(total_supply - total_demand) > 1e-6:
            return {
                "status": "UNBALANCED",
                "objective": float("inf"),
                "solve_time": time.time() - start_time,
                "iterations": 0,
                "n_nodes": self.n_nodes,
                "n_arcs": self.n_arcs,
                "error": f"Supply {total_supply} != Demand {total_demand}",
            }

        if total_supply == 0.0:  # 无需分配
            self._writeback_flow()
            return {
                "status": "OPTIMAL",
                "objective": 0.0,
                "solve_time": time.time() - start_time,
                "iterations": 0,
                "flows": self._build_flows_dataframe(),
                "total_flow": 0.0,
                "n_nodes": self.n_nodes,
                "n_arcs": self.n_arcs,
            }

        # 潜势（势）初始化为 0
        pi = np.zeros(self.n_nodes, dtype=float)
        iterations = 0

        # 主循环：直到所有供给清零
        def time_exceeded() -> bool:
            return (time_limit is not None) and ((time.time() - start_time) > float(time_limit))

        # 剩余供给节点与需求节点的快速查询
        def supply_nodes() -> List[int]:
            return [i for i, val in enumerate(self.b) if val > 1e-9]

        def demand_nodes() -> List[int]:
            return [i for i, val in enumerate(self.b) if val < -1e-9]

        status = "OPTIMAL"
        while True:
            if time_exceeded():
                status = "TIME_LIMIT"
                break

            S = supply_nodes()
            if not S:
                break  # 全部供给已分配

            s = S[-1]  # 任取一个有供给的源
            dist, prev = self._dijkstra(s, pi)

            # 在可达的需求节点中选取距离最小者
            T = demand_nodes()
            best_t = None
            best_dist = float("inf")
            for t in T:
                if prev[t] is not None and dist[t] < best_dist:
                    best_t = t
                    best_dist = dist[t]

            if best_t is None:
                # 从 s 无法到达任何仍有需求的节点 => 不可行
                status = "INFEASIBLE"
                break

            path = self._extract_path(prev, best_t)
            if not path:
                status = "INFEASIBLE"
                break

            # 本次可增广量：路径残量、s 的剩余供给、t 的剩余需求
            path_cap = self._path_residual_cap(path)
            delta = min(path_cap, self.b[s], -self.b[best_t])
            if delta <= 1e-12:
                status = "INFEASIBLE"
                break

            # 增广与供需更新
            self._augment(path, float(delta))
            self.b[s] -= float(delta)
            self.b[best_t] += float(delta)  # best_t < 0，+delta 使其趋近 0
            iterations += 1

            # 更新势（仅对本次可达的节点）
            reachable = np.isfinite(dist)
            pi[reachable] += dist[reachable]

        # 将残量网络回写到 self.flow
        self._writeback_flow()

        # 统计
        objective = float(np.dot(self.flow, self.arcs["cost"].astype(float).values))
        solve_time = time.time() - start_time
        flows_df = self._build_flows_dataframe()

        # 计算本次已成功送达的总流量：= 初始总供给 - 剩余供给
        delivered = float(total_supply - np.clip(self.b, 0, None).sum())

        # 最终一致性检查（仅在 OPTIMAL/TIME_LIMIT 下给出提示信息）
        imb = self._max_node_imbalance()

        result: Dict[str, Any] = {
            "status": status,
            "objective": objective,
            "solve_time": solve_time,
            "iterations": iterations,
            "flows": flows_df,
            "total_flow": delivered,
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
        }

        if status == "OPTIMAL":
            # 守恒必须满足
            if imb > 1e-7:
                result["status"] = "ERROR"
                result["error"] = f"node imbalance={imb:.3e} (should be ~0)"
        elif status in ("TIME_LIMIT", "INFEASIBLE"):
            result["error"] = f"status={status}, node imbalance={imb:.3e}"

        return result

    # ---- 输出 ----
    def _build_flows_dataframe(self) -> pd.DataFrame:
        flows_df = self.arcs.copy()
        flows_df["flow"] = self.flow
        return flows_df


# ========== I/O 与 CLI ==========

def _infer_graph_paths(
    nodes: Optional[str],
    arcs: Optional[str],
    meta: Optional[str],
    default_dir: str = "data/solver_graph",
) -> Tuple[str, str]:
    """
    自动发现图文件路径
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


def _read_table(path: str) -> pd.DataFrame:
    """读取表格文件"""
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


def solve_with_network_simplex(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    *,
    time_limit: Optional[float] = None,
    msg: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """使用 Successive Shortest Path 求解最小费用流"""
    if msg:
        print(f"[MinCostFlow] Solving: {len(nodes)} nodes, {len(arcs)} arcs")

    solver = NetworkSimplexSolver(nodes, arcs, msg=msg)
    result = solver.solve(time_limit=time_limit)

    # 成功或部分成功：返回实际流；失败则返回零流
    if result["status"] in ("OPTIMAL", "TIME_LIMIT"):
        flows_df = result["flows"]

        # 计算汇总统计
        total_cost = float((flows_df["flow"] * flows_df["cost"].astype(float)).sum())
        total_flow = float(result.get("total_flow", 0.0))

        # 按弧类型分组统计（兼容无 arc_id 的场景）
        by_type = {}
        if "arc_type" in flows_df.columns:
            tmp = flows_df.copy()
            tmp["_cost"] = tmp["flow"] * tmp["cost"].astype(float)
            g = tmp.groupby("arc_type", dropna=False).agg(
                flow_sum=("flow", "sum"),
                cost_sum=("_cost", "sum"),
            )
            # 组大小作为 arcs 数
            g["arcs"] = tmp.groupby("arc_type", dropna=False).size()
            for k, v in g.to_dict(orient="index").items():
                key = "nan" if pd.isna(k) else str(k)
                by_type[key] = {
                    "flow": float(v["flow_sum"]),
                    "cost": float(v["cost_sum"]),
                    "arcs": int(v["arcs"]),
                }

        summary = {
            "status": result["status"],
            "objective": float(result["objective"]),
            "total_cost": total_cost,
            "total_flow": total_flow,
            "nodes": int(result["n_nodes"]),
            "arcs": int(result["n_arcs"]),
            "solve_time": float(result["solve_time"]),
            "iterations": int(result["iterations"]),
            "by_type": by_type,
        }

        if msg:
            print(f"[MinCostFlow] Status: {summary['status']}")
            print(f"[MinCostFlow] Objective: {summary['objective']:.6g}")
            print(f"[MinCostFlow] Solve time: {summary['solve_time']:.3f}s")
            print(f"[MinCostFlow] Delivered flow: {total_flow:.6g}")

        return flows_df, summary

    else:
        # 失败：返回零流
        empty_flows = arcs.copy()
        empty_flows["flow"] = 0.0

        summary = {
            "status": result["status"],
            "objective": float(result.get("objective", float("inf"))),
            "total_cost": 0.0,
            "total_flow": 0.0,
            "nodes": int(result.get("n_nodes", len(nodes))),
            "arcs": int(result.get("n_arcs", len(arcs))),
            "solve_time": float(result.get("solve_time", 0.0)),
            "iterations": int(result.get("iterations", 0)),
            "error": result.get("error", "Unknown error"),
            "by_type": {},
        }

        if msg:
            print(f"[MinCostFlow] ERROR: {result['status']}")
            if "error" in result:
                print(f"[MinCostFlow] Error details: {result['error']}")

        return empty_flows, summary


def main():
    """主函数"""
    ap = argparse.ArgumentParser(description="最小费用流求解器（SSP 带势；默认读取 data/solver_graph/*）")
    ap.add_argument("--nodes", type=str, default=None, help="nodes 文件（parquet/csv），默认自动发现")
    ap.add_argument("--arcs", type=str, default=None, help="arcs 文件（parquet/csv），默认自动发现")
    ap.add_argument("--meta", type=str, default=None, help="meta.json 路径，默认 data/solver_graph/meta.json")
    ap.add_argument("--out", type=str, default="outputs", help="输出目录")
    ap.add_argument("--time-limit", type=int, default=3600, help="时间限制（秒）")
    ap.add_argument("--msg", type=int, default=1, help="求解日志（1/0）")
    ap.add_argument("--zero-tol", type=float, default=1e-9, help="将小于该阈值的流量置零")
    args = ap.parse_args()

    # 自动发现图文件
    nodes_path, arcs_path = _infer_graph_paths(args.nodes, args.arcs, args.meta)

    if args.msg:
        print(f"[MinCostFlow] Loading graph from:")
        print(f"  Nodes: {nodes_path}")
        print(f"  Arcs:  {arcs_path}")

    # 读取图数据
    nodes = _read_table(nodes_path)
    arcs = _read_table(arcs_path)

    # 求解
    flows_df, summary = solve_with_network_simplex(
        nodes, arcs,
        time_limit=float(args.time_limit),
        msg=bool(args.msg),
    )

    # 将小于阈值的流量置零
    flows_df.loc[flows_df["flow"].abs() < float(args.zero_tol), "flow"] = 0.0

    # 保存结果
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

    print("[MinCostFlow] done.")
    print(f"  status      = {summary['status']}")
    print(f"  objective   = {summary['objective']:.6g}")
    print(f"  total_cost  = {summary['total_cost']:.6g}")
    print(f"  total_flow  = {summary['total_flow']:.6g}")
    print(f"  solve_time  = {summary['solve_time']:.3f}s")
    print(f"  arcs output -> {flows_written}")
    print(f"  summary     -> {meta_path}")


if __name__ == "__main__":
    main()
