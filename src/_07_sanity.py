# _07_sanity.py
# 负环检测模块：使用Bellman-Ford算法检测图中的负成本环
# 用于在求解前验证图的合理性，避免零时长负奖励造成无限回路

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json


def check_negative_cycles(
    arcs_df: pd.DataFrame, 
    nodes_df: pd.DataFrame, 
    keep_node_ids: Optional[Set[int]] = None,
    sample: bool = False,
    sample_ratio: float = 0.1
) -> Dict:
    """
    使用Bellman-Ford算法检测负成本环。
    
    Args:
        arcs_df: 弧数据框，必须包含 ['from_node_id', 'to_node_id', 'cost']
        nodes_df: 节点数据框，必须包含 ['node_id']
        keep_node_ids: 要检查的节点集合，None表示检查所有节点
        sample: 是否对大图进行采样检查
        sample_ratio: 采样比例（当sample=True时）
    
    Returns:
        Dict包含检测结果：
        - 'has_negative_cycle': bool
        - 'cycle_path': List[int] (如果存在负环)
        - 'cycle_arcs': List[Dict] (负环涉及的弧信息)
        - 'stats': Dict (统计信息)
    """
    
    # 数据验证
    required_arc_cols = ['from_node_id', 'to_node_id', 'cost']
    required_node_cols = ['node_id']
    
    for col in required_arc_cols:
        if col not in arcs_df.columns:
            raise ValueError(f"弧数据框缺少必需列: {col}")
    
    for col in required_node_cols:
        if col not in nodes_df.columns:
            raise ValueError(f"节点数据框缺少必需列: {col}")
    
    # 过滤弧数据
    arcs = arcs_df[required_arc_cols].copy()
    arcs = arcs.dropna()
    
    # 过滤节点数据
    if keep_node_ids is not None:
        arcs = arcs[
            arcs['from_node_id'].isin(keep_node_ids) & 
            arcs['to_node_id'].isin(keep_node_ids)
        ].copy()
    
    # 采样（如果启用）
    if sample and len(arcs) > 1000:
        sample_size = max(100, int(len(arcs) * sample_ratio))
        arcs = arcs.sample(n=sample_size, random_state=42).copy()
    
    # 获取所有节点，过滤无效值
    from_nodes = arcs['from_node_id'].dropna().astype(int).tolist()
    to_nodes = arcs['to_node_id'].dropna().astype(int).tolist()
    all_nodes = set(from_nodes) | set(to_nodes)
    all_nodes = sorted(list(all_nodes))
    
    if len(all_nodes) == 0:
        return {
            'has_negative_cycle': False,
            'cycle_path': [],
            'cycle_arcs': [],
            'stats': {'nodes': 0, 'arcs': 0, 'sampled': sample}
        }
    
    # 构建邻接表
    adj = {node: [] for node in all_nodes}
    arc_info = {}  # 存储弧的详细信息
    
    for _, row in arcs.iterrows():
        # 跳过包含NaN的行
        if pd.isna(row['from_node_id']) or pd.isna(row['to_node_id']):
            continue
            
        from_node = int(row['from_node_id'])
        to_node = int(row['to_node_id'])
        cost = float(row['cost'])
        
        # 确保节点在邻接表中
        if from_node not in adj:
            adj[from_node] = []
        if to_node not in adj:
            adj[to_node] = []
            
        adj[from_node].append(to_node)
        arc_info[(from_node, to_node)] = {
            'cost': cost,
            'arc_id': row.get('arc_id', None),
            'arc_type': row.get('arc_type', None)
        }
    
    # Bellman-Ford算法
    n = len(all_nodes)
    dist = {node: float('inf') for node in all_nodes}
    pred = {node: None for node in all_nodes}
    
    # 创建超源节点，向所有节点连零成本边
    super_source = min(all_nodes) - 1
    dist[super_source] = 0.0
    adj[super_source] = []
    
    # 添加从超源到所有节点的零成本边
    for node in all_nodes:
        adj[super_source].append(node)
        arc_info[(super_source, node)] = {'cost': 0.0, 'arc_id': None, 'arc_type': 'super_source'}
    
    # V-1次松弛
    for _ in range(n):
        for from_node in [super_source] + all_nodes:
            if from_node not in dist or dist[from_node] == float('inf'):
                continue
                
            for to_node in adj.get(from_node, []):
                if to_node not in dist:
                    continue
                edge_cost = arc_info[(from_node, to_node)]['cost']
                if dist[from_node] + edge_cost < dist[to_node]:
                    dist[to_node] = dist[from_node] + edge_cost
                    pred[to_node] = from_node
    
    # 第V次检查：检测负环
    negative_cycle = None
    for from_node in [super_source] + all_nodes:
        if from_node not in dist or dist[from_node] == float('inf'):
            continue
            
        for to_node in adj.get(from_node, []):
            if to_node not in dist:
                continue
            edge_cost = arc_info[(from_node, to_node)]['cost']
            if dist[from_node] + edge_cost < dist[to_node]:
                # 发现负环，回溯找到环路径
                negative_cycle = _trace_negative_cycle(from_node, to_node, pred, all_nodes)
                break
        
        if negative_cycle is not None:
            break
    
    # 构建结果
    result = {
        'has_negative_cycle': negative_cycle is not None,
        'cycle_path': negative_cycle if negative_cycle else [],
        'cycle_arcs': [],
        'stats': {
            'nodes': len(all_nodes),
            'arcs': len(arcs),
            'sampled': sample,
            'sample_ratio': sample_ratio if sample else 1.0
        }
    }
    
    # 如果发现负环，提取环中的弧信息
    if negative_cycle:
        cycle_arcs = []
        for i in range(len(negative_cycle)):
            from_node = negative_cycle[i]
            to_node = negative_cycle[(i + 1) % len(negative_cycle)]
            
            if (from_node, to_node) in arc_info:
                arc_data = arc_info[(from_node, to_node)].copy()
                arc_data['from_node_id'] = from_node
                arc_data['to_node_id'] = to_node
                cycle_arcs.append(arc_data)
        
        result['cycle_arcs'] = cycle_arcs
    
    return result


def _trace_negative_cycle(start_node: int, end_node: int, pred: Dict[int, int], all_nodes: List[int]) -> List[int]:
    """回溯找到负环的路径。"""
    # 从end_node开始回溯，直到找到start_node或发现环
    visited = set()
    path = []
    current = end_node
    
    while current is not None and current not in visited:
        if current in all_nodes:  # 只包含图中的节点
            visited.add(current)
            path.append(current)
        current = pred.get(current)
        
        # 防止无限循环
        if len(path) > len(all_nodes):
            break
    
    # 如果找到了环，返回环部分
    if current in visited and current in all_nodes:
        cycle_start = path.index(current)
        return path[cycle_start:]
    
    return []


def export_negative_cycle_report(result: Dict, output_path: str) -> None:
    """导出负环检测报告。"""
    output_path = Path(output_path)
    
    if result['has_negative_cycle']:
        # 导出负环弧信息
        cycle_arcs_df = pd.DataFrame(result['cycle_arcs'])
        cycle_arcs_df.to_csv(output_path / 'negative_cycle_arcs.csv', index=False)
        
        # 导出检测报告
        report = {
            'detection_result': {
                'has_negative_cycle': result['has_negative_cycle'],
                'cycle_length': len(result['cycle_path']),
                'cycle_nodes': result['cycle_path']
            },
            'statistics': result['stats'],
            'cycle_arcs_count': len(result['cycle_arcs'])
        }
        
        with open(output_path / 'negative_cycle_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[sanity] 发现负环！已导出报告到: {output_path}")
        print(f"[sanity] 负环长度: {len(result['cycle_path'])}")
        print(f"[sanity] 涉及弧数: {len(result['cycle_arcs'])}")
    else:
        print(f"[sanity] 未发现负环 (检查了 {result['stats']['nodes']} 个节点, {result['stats']['arcs']} 条弧)")


def main():
    """命令行接口示例。"""
    import argparse
    
    parser = argparse.ArgumentParser(description='负环检测工具')
    parser.add_argument('--arcs', required=True, help='弧数据文件路径 (parquet/csv)')
    parser.add_argument('--nodes', required=True, help='节点数据文件路径 (parquet/csv)')
    parser.add_argument('--output', default='.', help='输出目录')
    parser.add_argument('--sample', action='store_true', help='启用采样检查')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='采样比例')
    
    args = parser.parse_args()
    
    # 读取数据
    if args.arcs.endswith('.parquet'):
        arcs_df = pd.read_parquet(args.arcs)
    else:
        arcs_df = pd.read_csv(args.arcs)
    
    if args.nodes.endswith('.parquet'):
        nodes_df = pd.read_parquet(args.nodes)
    else:
        nodes_df = pd.read_csv(args.nodes)
    
    # 执行检测
    result = check_negative_cycles(
        arcs_df, nodes_df, 
        sample=args.sample, 
        sample_ratio=args.sample_ratio
    )
    
    # 导出报告
    export_negative_cycle_report(result, args.output)


if __name__ == "__main__":
    main()
