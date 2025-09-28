#!/usr/bin/env python3
"""
将求解结果构建为记录节点、边、边成本的JSON图结构文件输出

该脚本读取求解器输出的parquet文件，构建包含以下信息的JSON图结构：
- nodes: 节点信息（节点ID、属性等）
- edges: 边信息（源节点、目标节点、流量、成本等）
- metadata: 图的基本信息和求解结果摘要

使用方法:
    python export_graph_to_json.py [--output OUTPUT_FILE] [--data-dir DATA_DIR]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def load_parquet_safe(file_path: Path) -> Optional[pd.DataFrame]:
    """安全加载parquet文件"""
    try:
        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            print(f"警告: 文件不存在 {file_path}")
            return None
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None


def load_json_safe(file_path: Path) -> Optional[Dict]:
    """安全加载JSON文件"""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"警告: 文件不存在 {file_path}")
            return None
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None


def build_nodes_json(nodes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """构建节点JSON结构"""
    nodes = []
    
    for _, row in nodes_df.iterrows():
        node = {
            "id": str(row.get('node_id', row.name)),
            "type": "node",  # 节点统一类型为node
            "attributes": {}
        }
        
        # 添加所有其他列作为属性
        for col in nodes_df.columns:
            if col not in ['node_id']:
                value = row[col]
                # 处理numpy类型
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                node["attributes"][col] = value
        
        nodes.append(node)
    
    return nodes


def build_edges_json(arcs_df: pd.DataFrame, flows_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """构建边JSON结构"""
    edges = []
    
    for _, row in arcs_df.iterrows():
        edge = {
            "id": str(row.get('arc_id', row.name)),
            "source": str(row.get('from_node_id', row.get('from_node', row.get('source', '')))),
            "target": str(row.get('to_node_id', row.get('to_node', row.get('target', '')))),
            "type": row.get('arc_type', 'unknown'),
            "cost": float(row.get('cost', 0.0)),
            "capacity": float(row.get('capacity', float('inf'))),
            "flow": 0.0,  # 默认流量为0
            "attributes": {}
        }
        
        # 如果有流量数据，添加流量信息
        if flows_df is not None:
            # 尝试匹配流量数据
            arc_id = row.get('arc_id', row.name)
            flow_row = flows_df[flows_df.get('arc_id', flows_df.index) == arc_id]
            if not flow_row.empty:
                edge["flow"] = float(flow_row.iloc[0].get('flow', 0.0))
        
        # 添加所有其他列作为属性
        for col in arcs_df.columns:
            if col not in ['arc_id', 'from_node_id', 'to_node_id', 'from_node', 'to_node', 'source', 'target', 'arc_type', 'type', 'cost', 'capacity']:
                value = row[col]
                # 处理numpy类型
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                edge["attributes"][col] = value
        
        edges.append(edge)
    
    return edges


def build_graph_json(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    flows_df: Optional[pd.DataFrame] = None,
    solve_summary: Optional[Dict] = None,
    meta_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """构建完整的图JSON结构"""
    
    # 构建节点和边
    nodes = build_nodes_json(nodes_df)
    edges = build_edges_json(arcs_df, flows_df)
    
    # 构建图结构
    graph = {
        "metadata": {
            "description": "Network-Based SAEV Operation Optimization Graph",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_flow": sum(edge["flow"] for edge in edges),
            "total_cost": sum(edge["cost"] * edge["flow"] for edge in edges),
            "created_at": pd.Timestamp.now().isoformat()
        },
        "nodes": nodes,
        "edges": edges
    }
    
    # 添加求解结果摘要
    if solve_summary:
        graph["metadata"]["solve_summary"] = solve_summary
    
    # 添加图元信息
    if meta_info:
        graph["metadata"]["graph_info"] = meta_info
    
    # 按类型统计边
    edge_types = {}
    for edge in edges:
        edge_type = edge["type"]
        if edge_type not in edge_types:
            edge_types[edge_type] = {
                "count": 0,
                "total_flow": 0.0,
                "total_cost": 0.0
            }
        edge_types[edge_type]["count"] += 1
        edge_types[edge_type]["total_flow"] += edge["flow"]
        edge_types[edge_type]["total_cost"] += edge["cost"] * edge["flow"]
    
    graph["metadata"]["edge_types"] = edge_types
    
    return graph


def main():
    parser = argparse.ArgumentParser(
        description="将求解结果构建为JSON图结构文件输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python export_graph_to_json.py
    python export_graph_to_json.py --output graph.json
    python export_graph_to_json.py --data-dir src/pipeline/data/solver_graph --output graph.json
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="graph_structure.json",
        help="输出JSON文件路径 (默认: graph_structure.json)"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="src/pipeline/data/solver_graph",
        help="数据目录路径 (默认: src/pipeline/data/solver_graph)"
    )
    
    parser.add_argument(
        "--flows-dir",
        type=str,
        default="src/pipeline/outputs",
        help="流量数据目录路径 (默认: src/pipeline/outputs)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    flows_dir = Path(args.flows_dir)
    output_file = Path(args.output)
    
    if args.verbose:
        print(f"数据目录: {data_dir}")
        print(f"流量目录: {flows_dir}")
        print(f"输出文件: {output_file}")
    
    # 检查数据目录
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        sys.exit(1)
    
    # 加载数据文件
    print("正在加载数据文件...")
    
    # 加载节点数据
    nodes_file = data_dir / "nodes.parquet"
    nodes_df = load_parquet_safe(nodes_file)
    if nodes_df is None:
        print("错误: 无法加载节点数据")
        sys.exit(1)
    
    # 加载边数据
    arcs_file = data_dir / "arcs.parquet"
    arcs_df = load_parquet_safe(arcs_file)
    if arcs_df is None:
        print("错误: 无法加载边数据")
        sys.exit(1)
    
    # 加载流量数据（可选）
    flows_file = flows_dir / "flows.parquet"
    flows_df = load_parquet_safe(flows_file)
    if flows_df is not None and args.verbose:
        print(f"已加载流量数据: {len(flows_df)} 条记录")
    
    # 加载求解摘要（可选）
    solve_summary_file = flows_dir / "solve_summary.json"
    solve_summary = load_json_safe(solve_summary_file)
    
    # 加载图元信息（可选）
    meta_file = data_dir / "meta.json"
    meta_info = load_json_safe(meta_file)
    
    print(f"节点数据: {len(nodes_df)} 个节点")
    print(f"边数据: {len(arcs_df)} 条边")
    
    # 构建图JSON结构
    print("正在构建图结构...")
    graph_json = build_graph_json(
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        flows_df=flows_df,
        solve_summary=solve_summary,
        meta_info=meta_info
    )
    
    # 保存JSON文件
    print(f"正在保存到 {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_json, f, ensure_ascii=False, indent=2)
        
        print(f"成功保存图结构到 {output_file}")
        print(f"包含 {len(graph_json['nodes'])} 个节点和 {len(graph_json['edges'])} 条边")
        
        if args.verbose:
            print("\n图结构摘要:")
            print(f"  总流量: {graph_json['metadata']['total_flow']}")
            print(f"  总成本: {graph_json['metadata']['total_cost']}")
            print(f"  边类型统计:")
            for edge_type, stats in graph_json['metadata']['edge_types'].items():
                print(f"    {edge_type}: {stats['count']} 条边, 流量 {stats['total_flow']:.2f}, 成本 {stats['total_cost']:.2f}")
    
    except Exception as e:
        print(f"错误: 无法保存文件 {output_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
