#!/usr/bin/env python3
"""
简化版图结构导出脚本
生成包含节点、边、边成本的简洁JSON图结构

使用方法:
    python export_simple_graph.py [--output OUTPUT_FILE]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def load_data(data_dir: Path, flows_dir: Path) -> tuple:
    """加载所有必要的数据文件"""
    # 加载节点数据
    nodes_file = data_dir / "nodes.parquet"
    if not nodes_file.exists():
        raise FileNotFoundError(f"节点文件不存在: {nodes_file}")
    nodes_df = pd.read_parquet(nodes_file)
    
    # 加载边数据
    arcs_file = data_dir / "arcs.parquet"
    if not arcs_file.exists():
        raise FileNotFoundError(f"边文件不存在: {arcs_file}")
    arcs_df = pd.read_parquet(arcs_file)
    
    # 加载流量数据
    flows_file = flows_dir / "flows.parquet"
    flows_df = None
    if flows_file.exists():
        flows_df = pd.read_parquet(flows_file)
    
    # 加载求解摘要
    solve_summary_file = flows_dir / "solve_summary.json"
    solve_summary = None
    if solve_summary_file.exists():
        with open(solve_summary_file, 'r', encoding='utf-8') as f:
            solve_summary = json.load(f)
    
    return nodes_df, arcs_df, flows_df, solve_summary


def create_simple_nodes(nodes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """创建简化的节点列表"""
    nodes = []
    
    for _, row in nodes_df.iterrows():
        node = {
            "id": str(row.get('node_id', row.name)),
            "type": "node"  # 节点统一类型为node
        }
        
        # 添加关键属性
        key_attrs = ['zone', 't', 'soc', 'supply']  # 根据实际列名调整
        for attr in key_attrs:
            if attr in row and pd.notna(row[attr]):
                if isinstance(row[attr], (np.integer, np.floating)):
                    node[attr] = row[attr].item()
                else:
                    node[attr] = str(row[attr])
        
        nodes.append(node)
    
    return nodes


def create_simple_edges(arcs_df: pd.DataFrame, flows_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """创建简化的边列表"""
    edges = []
    
    # 创建流量映射
    flow_map = {}
    if flows_df is not None:
        for _, row in flows_df.iterrows():
            arc_id = row.get('arc_id', row.name)
            flow = row.get('flow', 0.0)
            flow_map[arc_id] = float(flow)
    
    for _, row in arcs_df.iterrows():
        arc_id = row.get('arc_id', row.name)
        flow = flow_map.get(arc_id, 0.0)
        
        edge = {
            "id": str(arc_id),
            "source": str(row.get('from_node_id', row.get('from_node', row.get('source', '')))),
            "target": str(row.get('to_node_id', row.get('to_node', row.get('target', '')))),
            "type": row.get('arc_type', 'unknown'),
            "cost": float(row.get('cost', 0.0)),
            "flow": flow
        }
        
        # 添加容量信息（如果存在）
        if 'capacity' in row and pd.notna(row['capacity']):
            edge["capacity"] = float(row['capacity'])
        
        edges.append(edge)
    
    return edges


def create_graph_summary(nodes: List[Dict], edges: List[Dict], solve_summary: Optional[Dict] = None) -> Dict[str, Any]:
    """创建图摘要信息"""
    # 统计边类型
    edge_types = {}
    for edge in edges:
        edge_type = edge["type"]
        if edge_type not in edge_types:
            edge_types[edge_type] = {
                "count": 0,
                "total_flow": 0.0,
                "total_cost": 0.0,
                "capacities": [],
                "max_capacity": 0.0,
                "min_capacity": float('inf')
            }
        edge_types[edge_type]["count"] += 1
        edge_types[edge_type]["total_flow"] += edge["flow"]
        edge_types[edge_type]["total_cost"] += edge["cost"] * edge["flow"]
        
        # 统计容量信息
        if "capacity" in edge:
            capacity = edge["capacity"]
            edge_types[edge_type]["capacities"].append(capacity)
            edge_types[edge_type]["max_capacity"] = max(edge_types[edge_type]["max_capacity"], capacity)
            edge_types[edge_type]["min_capacity"] = min(edge_types[edge_type]["min_capacity"], capacity)
    
    # 清理容量统计
    for edge_type in edge_types:
        if edge_types[edge_type]["min_capacity"] == float('inf'):
            edge_types[edge_type]["min_capacity"] = 0.0
        # 去重容量值
        edge_types[edge_type]["unique_capacities"] = list(set(edge_types[edge_type]["capacities"]))
        del edge_types[edge_type]["capacities"]  # 删除原始列表以节省空间
    
    # 计算总流量和总成本
    total_flow = sum(edge["flow"] for edge in edges)
    total_cost = sum(edge["cost"] * edge["flow"] for edge in edges)
    
    summary = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_flow": total_flow,
        "total_cost": total_cost,
        "edge_types": edge_types
    }
    
    # 添加求解摘要
    if solve_summary:
        summary["solve_status"] = solve_summary.get("status", "unknown")
        summary["objective"] = solve_summary.get("objective", 0.0)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="导出简化的图结构JSON文件",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="simple_graph.json",
        help="输出JSON文件路径 (默认: simple_graph.json)"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="src/pipeline/data/solver_graph",
        help="数据目录路径"
    )
    
    parser.add_argument(
        "--flows-dir",
        type=str,
        default="src/pipeline/outputs",
        help="流量数据目录路径"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    try:
        # 设置路径
        data_dir = Path(args.data_dir)
        flows_dir = Path(args.flows_dir)
        output_file = Path(args.output)
        
        if args.verbose:
            print(f"数据目录: {data_dir}")
            print(f"流量目录: {flows_dir}")
            print(f"输出文件: {output_file}")
        
        # 加载数据
        print("正在加载数据...")
        nodes_df, arcs_df, flows_df, solve_summary = load_data(data_dir, flows_dir)
        
        print(f"节点数据: {len(nodes_df)} 个节点")
        print(f"边数据: {len(arcs_df)} 条边")
        if flows_df is not None:
            print(f"流量数据: {len(flows_df)} 条记录")
        
        # 创建简化的图结构
        print("正在构建图结构...")
        nodes = create_simple_nodes(nodes_df)
        edges = create_simple_edges(arcs_df, flows_df)
        summary = create_graph_summary(nodes, edges, solve_summary)
        
        # 构建最终JSON
        graph = {
            "metadata": {
                "description": "Network-Based SAEV Operation Optimization Graph (Simplified)",
                "created_at": pd.Timestamp.now().isoformat(),
                **summary
            },
            "nodes": nodes,
            "edges": edges
        }
        
        # 保存文件
        print(f"正在保存到 {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        
        print(f"成功保存图结构到 {output_file}")
        print(f"包含 {len(nodes)} 个节点和 {len(edges)} 条边")
        
        if args.verbose:
            print(f"\n图结构摘要:")
            print(f"  总流量: {summary['total_flow']}")
            print(f"  总成本: {summary['total_cost']}")
            print(f"  求解状态: {summary.get('solve_status', 'unknown')}")
            print(f"  边类型统计:")
            for edge_type, stats in summary['edge_types'].items():
                capacity_info = ""
                if 'unique_capacities' in stats and len(stats['unique_capacities']) > 0:
                    capacities = stats['unique_capacities']
                    if len(capacities) == 1:
                        cap_val = capacities[0]
                        if cap_val >= 1e12:
                            capacity_info = " (无限容量)"
                        else:
                            capacity_info = f" (容量: {cap_val})"
                    else:
                        capacity_info = f" (容量: {stats['min_capacity']}-{stats['max_capacity']})"
                
                print(f"    {edge_type}: {stats['count']} 条边, 流量 {stats['total_flow']:.2f}{capacity_info}")
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
