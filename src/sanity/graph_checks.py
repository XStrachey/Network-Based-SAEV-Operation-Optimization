# sanity/graph_checks.py
# 图连通性和安全性检查

from __future__ import annotations

import pandas as pd
import networkx as nx
from typing import Dict, Set, List, Tuple, Any
import numpy as np


def check_connectivity(
    arcs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    target_nodes: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    检查图的连通性
    
    Args:
        arcs_df: 弧数据框
        nodes_df: 节点数据框
        target_nodes: 目标节点集合（如服务闸门节点）
    
    Returns:
        连通性检查结果字典
    """
    if arcs_df.empty or nodes_df.empty:
        return {
            'connected_components': 0,
            'isolated_nodes': len(nodes_df),
            'connectivity_passed': False,
            'reachable_targets': 0,
            'total_targets': len(target_nodes) if target_nodes else 0
        }
    
    # 构建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    for _, node in nodes_df.iterrows():
        G.add_node(int(node['node_id']))
    
    # 添加边
    for _, arc in arcs_df.iterrows():
        from_node = int(arc['from_node_id'])
        to_node = int(arc['to_node_id'])
        if from_node != to_node:  # 避免自环
            G.add_edge(from_node, to_node)
    
    # 计算连通分量
    connected_components = list(nx.weakly_connected_components(G))
    
    # 找出孤立节点
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    
    # 检查目标节点可达性
    reachable_targets = 0
    if target_nodes:
        reachable_nodes = set()
        for component in connected_components:
            if len(component) > 1:  # 非单节点分量
                reachable_nodes.update(component)
        
        reachable_targets = len(target_nodes & reachable_nodes)
    
    return {
        'connected_components': len(connected_components),
        'isolated_nodes': len(isolated_nodes),
        'connectivity_passed': len(isolated_nodes) == 0 and len(connected_components) <= 1,
        'reachable_targets': reachable_targets,
        'total_targets': len(target_nodes) if target_nodes else 0,
        'component_sizes': [len(comp) for comp in connected_components]
    }


def check_degree_constraints(
    arcs_df: pd.DataFrame,
    min_outdeg: int = 1,
    min_indeg: int = 1
) -> Dict[str, Any]:
    """
    检查节点度约束
    
    Args:
        arcs_df: 弧数据框
        min_outdeg: 最小出度
        min_indeg: 最小入度
    
    Returns:
        度约束检查结果
    """
    if arcs_df.empty:
        return {
            'outdeg_violations': 0,
            'indeg_violations': 0,
            'violation_nodes': [],
            'passed': True
        }
    
    # 计算出入度
    outdeg = arcs_df.groupby('from_node_id').size().to_dict()
    indeg = arcs_df.groupby('to_node_id').size().to_dict()
    
    # 找出所有节点
    all_nodes = set(arcs_df['from_node_id']) | set(arcs_df['to_node_id'])
    
    outdeg_violations = []
    indeg_violations = []
    
    for node in all_nodes:
        out_count = outdeg.get(node, 0)
        in_count = indeg.get(node, 0)
        
        if out_count < min_outdeg:
            outdeg_violations.append((node, out_count, min_outdeg))
        
        if in_count < min_indeg:
            indeg_violations.append((node, in_count, min_indeg))
    
    return {
        'outdeg_violations': len(outdeg_violations),
        'indeg_violations': len(indeg_violations),
        'violation_nodes': outdeg_violations + indeg_violations,
        'passed': len(outdeg_violations) == 0 and len(indeg_violations) == 0
    }


def check_reposition_arc_directionality(
    repos_df: pd.DataFrame,
    sample_size: int = 500
) -> Dict[str, Any]:
    """
    检查重定位弧的方向性是否正确
    
    Args:
        repos_df: 重定位弧数据框
        sample_size: 采样大小
    
    Returns:
        方向性检查结果
    """
    if repos_df.empty or 'delta_dp' not in repos_df.columns:
        return {
            'sampled_arcs': 0,
            'correct_directions': 0,
            'incorrect_directions': 0,
            'direction_accuracy': 1.0
        }
    
    # 采样检查
    if len(repos_df) > sample_size:
        sample_df = repos_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = repos_df
    
    correct_count = 0
    total_count = len(sample_df)
    
    for _, arc in sample_df.iterrows():
        delta_dp = arc['delta_dp']
        from_zone = arc['i']
        to_zone = arc['j']
        
        # 检查方向性：保留的弧应该是朝压力更高的方向
        if delta_dp > 0:
            # 应该保留 i->j 方向
            if from_zone < to_zone:  # 简化判断：假设保留的是 i<j 的方向
                correct_count += 1
        elif delta_dp < 0:
            # 应该保留 j->i 方向
            if from_zone > to_zone:  # 简化判断：假设保留的是 i>j 的方向
                correct_count += 1
        else:
            # delta_dp = 0，两个方向都应该保留，算作正确
            correct_count += 1
    
    return {
        'sampled_arcs': total_count,
        'correct_directions': correct_count,
        'incorrect_directions': total_count - correct_count,
        'direction_accuracy': correct_count / total_count if total_count > 0 else 1.0
    }


def comprehensive_graph_check(
    arcs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    svc_gates_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    综合图检查
    
    Args:
        arcs_df: 弧数据框
        nodes_df: 节点数据框
        svc_gates_df: 服务闸门弧数据框
    
    Returns:
        综合检查结果
    """
    # 分离重定位弧
    repos_df = arcs_df[arcs_df['arc_type'] == 'reposition'] if 'arc_type' in arcs_df.columns else pd.DataFrame()
    
    # 连通性检查
    target_nodes = None
    if svc_gates_df is not None and not svc_gates_df.empty:
        target_nodes = set(svc_gates_df['from_node_id']) | set(svc_gates_df['to_node_id'])
    
    connectivity_result = check_connectivity(arcs_df, nodes_df, target_nodes)
    
    # 度约束检查
    degree_result = check_degree_constraints(arcs_df)
    
    # 重定位弧方向性检查
    directionality_result = check_reposition_arc_directionality(repos_df)
    
    return {
        'connectivity': connectivity_result,
        'degree_constraints': degree_result,
        'directionality': directionality_result,
        'overall_passed': (
            connectivity_result['connectivity_passed'] and
            degree_result['passed'] and
            directionality_result['direction_accuracy'] >= 0.9
        )
    }
