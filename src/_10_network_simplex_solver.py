# _10_network_simplex_solver.py
# 使用Network Simplex算法求解最小费用流问题
# 输入：_08_build_solver_graph.py生成的图数据
# 输出：flows.parquet和solve_summary.json

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from collections import deque

# 导入配置
from _01_network_config import get_network_config


class NetworkSimplexSolver:
    """Network Simplex求解器"""
    
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        self.nodes = nodes.copy()
        self.arcs = arcs.copy()
        self.n_nodes = len(nodes)
        self.n_arcs = len(arcs)
        
        # 初始化流量
        self.flow = np.zeros(self.n_arcs)
        
    def solve(self, max_iterations: int = 1000) -> Dict:
        """求解最小费用流"""
        start_time = time.time()
        
        try:
            # 检查供需平衡
            total_supply = self.nodes['supply'].clip(lower=0).sum()
            total_demand = -self.nodes['supply'].clip(upper=0).sum()
            
            if abs(total_supply - total_demand) > 1e-6:
                return {
                    'status': 'UNBALANCED',
                    'objective': float('inf'),
                    'solve_time': time.time() - start_time,
                    'iterations': 0,
                    'error': f'Supply {total_supply} != Demand {total_demand}'
                }
            
            # 寻找初始可行解
            if not self._find_feasible_solution():
                return {
                    'status': 'INFEASIBLE',
                    'objective': float('inf'),
                    'solve_time': time.time() - start_time,
                    'iterations': 0,
                    'error': 'No feasible solution found'
                }
            
            # 计算目标函数
            objective = self._compute_objective()
            solve_time = time.time() - start_time
            
            # 构建结果
            flows_df = self._build_flows_dataframe()
            
            return {
                'status': 'OPTIMAL',
                'objective': objective,
                'solve_time': solve_time,
                'iterations': 1,  # 简化版本只进行一次迭代
                'flows': flows_df,
                'total_flow': flows_df['flow'].sum(),
                'n_nodes': self.n_nodes,
                'n_arcs': self.n_arcs
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'objective': float('inf'),
                'solve_time': time.time() - start_time,
                'iterations': 0,
                'error': str(e)
            }
    
    def _find_feasible_solution(self) -> bool:
        """寻找初始可行解 - 使用改进的最大流方法"""
        try:
            # 找到源点和汇点
            sources = self.nodes[self.nodes['supply'] > 0]['node_id'].tolist()
            sinks = self.nodes[self.nodes['supply'] < 0]['node_id'].tolist()
            
            if not sources or not sinks:
                return True  # 没有供需，直接可行
            
            # 构建邻接表
            outgoing = {}
            for idx, arc in self.arcs.iterrows():
                u = arc['from_node_id']
                if u not in outgoing:
                    outgoing[u] = []
                outgoing[u].append((arc['to_node_id'], idx))
            
            # 对每个源点，尝试发送流量到汇点
            for source in sources:
                supply = self.nodes[self.nodes['node_id'] == source]['supply'].iloc[0]
                remaining_supply = supply
                
                # 尝试多次发送，直到供给用完或无法找到路径
                max_attempts = 100  # 防止无限循环
                attempts = 0
                
                while remaining_supply > 1e-6 and attempts < max_attempts:
                    attempts += 1
                    
                    # 寻找从源到任意汇点的路径
                    path = self._find_path_to_sink(source, sinks, outgoing)
                    
                    if not path:
                        break  # 无法找到路径
                    
                    # 计算路径上的最小剩余容量
                    min_capacity = float('inf')
                    for arc_idx in path:
                        arc_capacity = self.arcs.iloc[arc_idx]['capacity']
                        remaining_capacity = arc_capacity - self.flow[arc_idx]
                        min_capacity = min(min_capacity, remaining_capacity)
                    
                    if min_capacity <= 1e-6:
                        break  # 路径已饱和
                    
                    # 发送流量
                    flow_amount = min(remaining_supply, min_capacity)
                    
                    for arc_idx in path:
                        self.flow[arc_idx] += flow_amount
                    
                    remaining_supply -= flow_amount
                
                if remaining_supply > 1e-6:
                    print(f"Warning: Could not send all supply from source {source}, remaining: {remaining_supply}")
            
            return True
            
        except Exception as e:
            print(f"Error finding feasible solution: {e}")
            return False
    
    def _find_path_to_sink(self, source: int, sinks: list, outgoing: dict) -> Optional[list]:
        """寻找从源到汇点的路径（BFS）"""
        queue = deque([(source, [])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current in sinks:
                return path
            
            for neighbor, arc_idx in outgoing.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # 检查弧是否还有剩余容量
                    arc_capacity = self.arcs.iloc[arc_idx]['capacity']
                    if self.flow[arc_idx] < arc_capacity - 1e-6:
                        queue.append((neighbor, path + [arc_idx]))
        
        return None
    
    def _compute_objective(self) -> float:
        """计算目标函数值"""
        return float(np.sum(self.flow * self.arcs['cost'].values))
    
    def _build_flows_dataframe(self) -> pd.DataFrame:
        """构建流量DataFrame"""
        flows_df = self.arcs.copy()
        flows_df['flow'] = self.flow
        return flows_df


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
    max_iterations: int = 1000,
    msg: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """使用Network Simplex求解最小费用流"""
    
    if msg:
        print(f"[Network Simplex] Solving: {len(nodes)} nodes, {len(arcs)} arcs")
    
    solver = NetworkSimplexSolver(nodes, arcs)
    result = solver.solve(max_iterations=max_iterations)
    
    if result['status'] == 'OPTIMAL':
        flows_df = result['flows']
        
        # 计算汇总统计
        total_cost = float((flows_df['flow'] * flows_df['cost']).sum())
        total_flow = float(flows_df['flow'].sum())
        
        # 按弧类型分组统计
        by_type = {}
        if "arc_type" in flows_df.columns:
            tmp = flows_df.copy()
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
            "status": result['status'],
            "objective": result['objective'],
            "total_cost": total_cost,
            "total_flow": total_flow,
            "nodes": int(result['n_nodes']),
            "arcs": int(result['n_arcs']),
            "solve_time": result['solve_time'],
            "iterations": result['iterations'],
            "by_type": by_type,
        }
        
        if msg:
            print(f"[Network Simplex] Status: {result['status']}")
            print(f"[Network Simplex] Objective: {result['objective']:.6g}")
            print(f"[Network Simplex] Solve time: {result['solve_time']:.3f}s")
            print(f"[Network Simplex] Total flow: {total_flow:.1f}")
        
        return flows_df, summary
    else:
        # 返回空结果
        empty_flows = arcs.copy()
        empty_flows['flow'] = 0.0
        
        summary = {
            "status": result['status'],
            "objective": result.get('objective', float('inf')),
            "total_cost": 0.0,
            "total_flow": 0.0,
            "nodes": int(result['n_nodes']),
            "arcs": int(result['n_arcs']),
            "solve_time": result['solve_time'],
            "iterations": result['iterations'],
            "error": result.get('error', 'Unknown error'),
            "by_type": {},
        }
        
        if msg:
            print(f"[Network Simplex] ERROR: {result['status']}")
            if 'error' in result:
                print(f"[Network Simplex] Error details: {result['error']}")
        
        return empty_flows, summary


def main():
    """主函数"""
    ap = argparse.ArgumentParser(description="Network Simplex求解器（默认读取 data/solver_graph/*）")
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
        print(f"[Network Simplex] Loading graph from:")
        print(f"  Nodes: {nodes_path}")
        print(f"  Arcs: {arcs_path}")
    
    # 读取图数据
    nodes = _read_table(nodes_path)
    arcs = _read_table(arcs_path)

    # 求解
    flows_df, summary = solve_with_network_simplex(
        nodes, arcs,
        max_iterations=10000,
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

    print("[Network Simplex] done.")
    print(f"  status      = {summary['status']}")
    print(f"  objective   = {summary['objective']:.6g}")
    print(f"  total_cost  = {summary['total_cost']:.6g}")
    print(f"  total_flow  = {summary['total_flow']:.6g}")
    print(f"  solve_time  = {summary['solve_time']:.3f}s")
    print(f"  arcs output -> {flows_written}")
    print(f"  summary     -> {meta_path}")


if __name__ == "__main__":
    main()
