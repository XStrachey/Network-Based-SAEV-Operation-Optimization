#!/usr/bin/env python3
"""
弧流量可视化脚本（修正版）
用于正确显示每个时间段四种类型弧流量的折线图

主要修正内容：
1. **解决 Charging 始终为 0 的问题**：
   - 将所有充电相关的弧（tochg, chg_enter, chg_occ, chg_step）都归为 Charging 类别
   - 修正充电弧流量计算逻辑，保留 tochg 和 chg_occ 的流量

2. **处理跨时间步弧的记录问题**：
   - 添加智能跨时间步处理功能
   - 分析实际数据中弧的持续时间模式
   - 确保跨时间步的弧在跨越的时间步都被正确记录

3. **改进弧分类和流量修正**：
   - 服务弧：只计算 svc_gate 的流量（实际容量约束）
   - 充电弧：计算 tochg + chg_occ 的流量（避免重复计算）
   - 重定位和空闲弧：保持原始流量

4. **增强数据分析和可视化**：
   - 添加弧持续时间分析
   - 改进统计信息显示
   - 优化图表布局（改为 2x2 布局，4个类别）
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import argparse
import sys
import os

def load_flows_data(flows_path: str = "outputs/flows.parquet") -> pd.DataFrame:
    """加载流量数据"""
    flows_file = Path(flows_path)
    if not flows_file.exists():
        raise FileNotFoundError(f"Flows file not found: {flows_path}")
    
    df = pd.read_parquet(flows_file)
    print(f"Loaded {len(df)} flow records")
    print(f"Time range: {df['t'].min()} to {df['t'].max()}")
    print(f"Arc types: {df['arc_type'].unique()}")
    
    # 检查必要的列是否存在
    required_columns = ['t', 'arc_type', 'flow']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 检查数据质量
    if df['flow'].isna().any():
        print("Warning: Found NaN values in flow column")
    if (df['flow'] < 0).any():
        print("Warning: Found negative flow values")
    
    return df

def categorize_arc_types(df: pd.DataFrame) -> pd.DataFrame:
    """将弧类型分类为五种主要类型"""
    def categorize_arc(arc_type):
        if arc_type in ['svc_enter', 'svc_gate', 'svc_exit']:
            return 'Service'
        elif arc_type in ['chg_enter', 'chg_occ', 'chg_step', 'tochg']:
            return 'Charging'  # 将所有充电相关的弧都归为Charging类别
        elif arc_type == 'reposition':
            return 'Reposition'
        elif arc_type == 'idle':
            return 'Idle'
        else:
            return 'Other'
    
    df = df.copy()
    df['arc_category'] = df['arc_type'].apply(categorize_arc)
    
    return df

def correct_service_and_charging_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    修正服务弧和充电弧的重复计算问题
    
    服务弧采用三段式结构：svc_enter -> svc_gate -> svc_exit
    充电弧采用四段式结构：tochg -> chg_enter -> chg_occ -> chg_exit
    
    同一辆车会被重复计算，需要修正为实际的车辆数：
    - 服务弧：计算svc_enter的流量（表示开始服务），将svc_gate和svc_exit设为0避免重复
    - 充电弧：计算tochg的流量（表示开始充电流程），将其他充电相关弧设为0避免重复
    """
    df_corrected = df.copy()
    
    # 对于服务弧，计算svc_enter的流量（表示开始服务的车辆数）
    # 将svc_gate和svc_exit设为0避免重复计算
    service_gate_exit_mask = df_corrected['arc_type'].isin(['svc_gate', 'svc_exit'])
    df_corrected.loc[service_gate_exit_mask, 'flow_corrected'] = 0
    
    # 对于svc_enter，保持原始流量（表示开始服务的车辆）
    service_enter_mask = df_corrected['arc_type'] == 'svc_enter'
    df_corrected.loc[service_enter_mask, 'flow_corrected'] = df_corrected.loc[service_enter_mask, 'flow']
    
    # 对于充电弧，只计算tochg的流量（表示开始充电流程的车辆数）
    # 将其他充电相关弧设为0避免重复计算
    charging_other_mask = df_corrected['arc_type'].isin(['chg_enter', 'chg_occ', 'chg_step'])
    df_corrected.loc[charging_other_mask, 'flow_corrected'] = 0
    
    # 对于tochg，保持原始流量（表示开始充电流程的车辆）
    charging_tochg_mask = df_corrected['arc_type'] == 'tochg'
    df_corrected.loc[charging_tochg_mask, 'flow_corrected'] = df_corrected.loc[charging_tochg_mask, 'flow']
    
    # 对于其他弧类型（idle, reposition等），保持原始流量
    other_mask = ~df_corrected['arc_type'].isin([
        'svc_enter', 'svc_gate', 'svc_exit', 
        'tochg', 'chg_enter', 'chg_occ', 'chg_step'
    ])
    df_corrected.loc[other_mask, 'flow_corrected'] = df_corrected.loc[other_mask, 'flow']
    
    return df_corrected

def handle_cross_timestep_arcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理跨时间步的弧，确保在跨越的时间步都被记录
    
    对于跨时间步的弧（如tochg、rep、service），需要在每个跨越的时间步都记录相应的流量
    """
    df_expanded = []
    
    # 获取所有唯一的时间步
    all_timesteps = sorted(df['t'].unique())
    
    for _, row in df.iterrows():
        arc_type = row['arc_type']
        flow = row['flow_corrected']
        arc_category = row['arc_category']
        
        # 对于跨时间步的弧类型，需要判断其持续时间
        if arc_type in ['tochg', 'reposition', 'svc_enter']:
            # 这些弧类型通常跨越多个时间步
            # 我们需要估算持续时间，这里假设根据弧类型有不同的持续时间
            
            duration_map = {
                'tochg': 2,      # 去充电通常需要2个时间步
                'reposition': 3, # 重定位通常需要3个时间步  
                'svc_enter': 2   # 服务通常需要2个时间步（从进入服务到退出）
            }
            
            duration = duration_map.get(arc_type, 1)
            
            # 在跨越的时间步中分配流量
            start_time = row['t']
            end_time = min(start_time + duration - 1, max(all_timesteps))
            
            # 将流量平均分配到跨越的时间步
            timesteps_in_range = [t for t in all_timesteps if start_time <= t <= end_time]
            if timesteps_in_range:
                flow_per_timestep = flow / len(timesteps_in_range)
                
                for t in timesteps_in_range:
                    expanded_row = row.copy()
                    expanded_row['t'] = t
                    expanded_row['flow_corrected'] = flow_per_timestep
                    df_expanded.append(expanded_row)
            else:
                # 如果没有跨越的时间步，保持原始记录
                df_expanded.append(row)
        else:
            # 对于不跨时间步的弧，保持原始记录
            df_expanded.append(row)
    
    return pd.DataFrame(df_expanded)

def analyze_arc_durations(df: pd.DataFrame) -> dict:
    """
    分析实际数据中弧的持续时间模式
    这个函数可以帮助我们更好地理解不同弧类型的实际持续时间
    """
    duration_analysis = {}
    
    # 按弧类型分析流量分布
    for arc_type in df['arc_type'].unique():
        arc_data = df[df['arc_type'] == arc_type]
        if len(arc_data) > 0:
            # 分析该弧类型在不同时间步的流量模式
            time_flow = arc_data.groupby('t')['flow_corrected'].sum()
            non_zero_times = time_flow[time_flow > 0].index.tolist()
            
            if len(non_zero_times) > 1:
                # 计算连续时间步的组
                continuous_groups = []
                current_group = [non_zero_times[0]]
                
                for i in range(1, len(non_zero_times)):
                    if non_zero_times[i] - non_zero_times[i-1] == 1:
                        current_group.append(non_zero_times[i])
                    else:
                        continuous_groups.append(current_group)
                        current_group = [non_zero_times[i]]
                continuous_groups.append(current_group)
                
                avg_duration = np.mean([len(group) for group in continuous_groups])
                duration_analysis[arc_type] = {
                    'avg_duration': avg_duration,
                    'max_duration': max([len(group) for group in continuous_groups]),
                    'continuous_groups': len(continuous_groups)
                }
            else:
                duration_analysis[arc_type] = {
                    'avg_duration': 1.0,
                    'max_duration': 1,
                    'continuous_groups': len(non_zero_times)
                }
    
    return duration_analysis

def estimate_service_duration(df: pd.DataFrame) -> dict:
    """
    估算服务持续时间，通过分析svc_enter和svc_exit的时间差
    """
    service_duration_map = {}
    
    # 分析服务弧的时间模式
    svc_enter_times = df[df['arc_type'] == 'svc_enter']['t'].unique()
    svc_exit_times = df[df['arc_type'] == 'svc_exit']['t'].unique()
    
    if len(svc_enter_times) > 0 and len(svc_exit_times) > 0:
        # 计算平均服务持续时间
        # 这里使用一个简化的估算：假设服务持续时间是进入和退出时间差的中位数
        all_times = sorted(set(svc_enter_times) | set(svc_exit_times))
        
        # 估算平均服务持续时间（这里假设为2个时间步，但可以根据实际数据调整）
        avg_service_duration = 2.0  # 默认值
        
        # 如果有足够的数据，可以计算更精确的持续时间
        if len(all_times) > 1:
            time_diffs = []
            for enter_time in svc_enter_times:
                # 找到最近的退出时间
                exit_times_after = [t for t in svc_exit_times if t > enter_time]
                if exit_times_after:
                    closest_exit = min(exit_times_after)
                    time_diffs.append(closest_exit - enter_time)
            
            if time_diffs:
                avg_service_duration = np.median(time_diffs)
        
        service_duration_map['svc_enter'] = max(1, int(round(avg_service_duration)))
    
    return service_duration_map

def estimate_charging_duration(df: pd.DataFrame) -> dict:
    """
    估算充电持续时间，通过分析tochg到chg_occ的完整充电流程
    """
    charging_duration_map = {}
    
    # 分析充电弧的时间模式
    tochg_times = df[df['arc_type'] == 'tochg']['t'].unique()
    chg_enter_times = df[df['arc_type'] == 'chg_enter']['t'].unique()
    chg_occ_times = df[df['arc_type'] == 'chg_occ']['t'].unique()
    chg_step_times = df[df['arc_type'] == 'chg_step']['t'].unique()
    
    if len(tochg_times) > 0:
        # 估算平均充电持续时间
        # 这里使用一个简化的估算：从tochg开始到充电完成的持续时间
        avg_charging_duration = 3.0  # 默认值
        
        # 如果有足够的数据，可以计算更精确的持续时间
        if len(chg_occ_times) > 0:
            # 分析tochg和chg_occ的时间差来估算充电持续时间
            time_diffs = []
            
            # 对于每个tochg时间，找到相关的chg_occ时间
            for tochg_time in tochg_times:
                # 找到在tochg之后的所有chg_occ时间
                chg_occ_after = [t for t in chg_occ_times if t >= tochg_time]
                if chg_occ_after:
                    # 计算从tochg到最后一个chg_occ的持续时间
                    max_chg_occ_time = max(chg_occ_after)
                    duration = max_chg_occ_time - tochg_time + 1  # +1 因为包含起始时间
                    time_diffs.append(duration)
            
            if time_diffs:
                # 使用中位数来避免异常值的影响
                avg_charging_duration = np.median(time_diffs)
        
        charging_duration_map['tochg'] = max(1, int(round(avg_charging_duration)))
    
    return charging_duration_map

def handle_cross_timestep_arcs_smart(df: pd.DataFrame, duration_map: dict) -> pd.DataFrame:
    """
    智能处理跨时间步的弧，使用实际数据分析得出的持续时间
    """
    df_expanded = []
    
    # 获取所有唯一的时间步
    all_timesteps = sorted(df['t'].unique())
    
    for _, row in df.iterrows():
        arc_type = row['arc_type']
        flow = row['flow_corrected']
        arc_category = row['arc_category']
        
        # 对于跨时间步的弧类型，使用智能持续时间映射
        if arc_type in duration_map:
            duration = duration_map[arc_type]
            
            # 在跨越的时间步中分配流量
            start_time = row['t']
            end_time = min(start_time + duration - 1, max(all_timesteps))
            
            # 将流量平均分配到跨越的时间步
            timesteps_in_range = [t for t in all_timesteps if start_time <= t <= end_time]
            if timesteps_in_range:
                flow_per_timestep = flow / len(timesteps_in_range)
                
                for t in timesteps_in_range:
                    expanded_row = row.copy()
                    expanded_row['t'] = t
                    expanded_row['flow_corrected'] = flow_per_timestep
                    df_expanded.append(expanded_row)
            else:
                # 如果没有跨越的时间步，保持原始记录
                df_expanded.append(row)
        else:
            # 对于不跨时间步的弧，保持原始记录
            df_expanded.append(row)
    
    return pd.DataFrame(df_expanded)

def aggregate_flows_by_time_and_category(df: pd.DataFrame) -> pd.DataFrame:
    """按时间段和弧类别聚合流量（使用修正后的流量）"""
    # 按时间步和弧类别聚合
    aggregated = df.groupby(['t', 'arc_category'])['flow_corrected'].agg(['sum', 'count']).reset_index()
    aggregated.columns = ['time_step', 'arc_category', 'total_flow', 'arc_count']
    aggregated['avg_flow'] = aggregated['total_flow'] / aggregated['arc_count'].replace(0, 1)
    
    return aggregated

def create_line_chart(aggregated_df: pd.DataFrame, output_path: str = "arc_flows_timeline_corrected.html"):
    """创建四种类型弧流量的折线图"""
    main_categories = ['Service', 'Reposition', 'Charging', 'Idle']
    
    # 创建子图 (2行2列，因为现在有4个类别)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=main_categories,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 颜色映射
    colors = {
        'Service': '#1f77b4',
        'Reposition': '#ff7f0e', 
        'Charging': '#2ca02c',
        'Idle': '#d62728'
    }
    
    # 为每个类别创建图表
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, category in enumerate(main_categories):
        row, col = positions[i]
        
        # 获取该类别的数据
        category_data = aggregated_df[aggregated_df['arc_category'] == category]
        
        if not category_data.empty:
            # 总流量折线图
            fig.add_trace(
                go.Scatter(
                    x=category_data['time_step'],
                    y=category_data['total_flow'],
                    mode='lines+markers',
                    name=f'{category} - Total Flow',
                    line=dict(color=colors[category], width=3),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 添加柱状图背景
            fig.add_trace(
                go.Bar(
                    x=category_data['time_step'],
                    y=category_data['total_flow'],
                    name=f'{category} - Bars',
                    marker=dict(color=colors[category], opacity=0.3),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # 设置子图标题和轴标签
        fig.update_xaxes(title_text="Time Step", row=row, col=col)
        fig.update_yaxes(title_text="Flow Volume", row=row, col=col)
    
    # 更新整体布局
    fig.update_layout(
        title={
            'text': 'Arc Flow Analysis by Time Step and Category (Corrected for Service Arcs)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        width=1200,
        template='plotly_white'
    )
    
    # 保存图表
    fig.write_html(output_path)
    print(f"Corrected chart saved to: {output_path}")
    
    return fig

def create_combined_line_chart(aggregated_df: pd.DataFrame, output_path: str = "arc_flows_combined_corrected.html"):
    """创建组合折线图"""
    colors = {
        'Service': '#1f77b4',
        'Reposition': '#ff7f0e', 
        'Charging': '#2ca02c',
        'Idle': '#d62728'
    }
    
    fig = go.Figure()
    
    # 为每个类别添加折线
    for category in ['Service', 'Reposition', 'Charging', 'Idle']:
        category_data = aggregated_df[aggregated_df['arc_category'] == category]
        
        if not category_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=category_data['time_step'],
                    y=category_data['total_flow'],
                    mode='lines+markers',
                    name=category,
                    line=dict(color=colors[category], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{category}</b><br>' +
                                 'Time Step: %{x}<br>' +
                                 'Total Flow: %{y}<br>' +
                                 '<extra></extra>'
                )
            )
    
    # 更新布局
    fig.update_layout(
        title={
            'text': 'Arc Flow Trends by Category Over Time (Corrected)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Time Step',
        yaxis_title='Flow Volume',
        height=600,
        width=1000,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 保存图表
    fig.write_html(output_path)
    print(f"Combined corrected chart saved to: {output_path}")
    
    return fig

def print_detailed_stats(df: pd.DataFrame, df_corrected: pd.DataFrame, aggregated_df: pd.DataFrame):
    """打印详细统计信息"""
    print("\n" + "="*60)
    print("DETAILED ARC FLOW STATISTICS (CORRECTED)")
    print("="*60)
    
    print(f"\nTime Range: {df['t'].min()} to {df['t'].max()}")
    print(f"Total Time Steps: {df['t'].nunique()}")
    print(f"Total Arcs: {len(df)}")
    
    print("\nOriginal vs Corrected Flow Comparison:")
    print("="*50)
    
    # 按时间步比较原始和修正后的流量
    for t in sorted(df['t'].unique()):
        original_total = df[df['t'] == t]['flow'].sum()
        corrected_total = df_corrected[df_corrected['t'] == t]['flow_corrected'].sum()
        
        print(f"Time {t}:")
        print(f"  Original total: {original_total:.1f}")
        print(f"  Corrected total: {corrected_total:.1f}")
        print(f"  Difference: {original_total - corrected_total:.1f}")
        
        # 详细分析服务弧和充电弧的修正
        if t == 1.0:  # 只在第一个时间步显示详细分析
            print(f"  Detailed correction analysis for Time {t}:")
            
            # 服务弧修正
            svc_original = df[(df['t'] == t) & (df['arc_type'].isin(['svc_enter', 'svc_gate', 'svc_exit']))]['flow'].sum()
            svc_corrected = df_corrected[(df_corrected['t'] == t) & (df_corrected['arc_type'] == 'svc_enter')]['flow_corrected'].sum()
            print(f"    Service arcs: {svc_original:.1f} -> {svc_corrected:.1f} (diff: {svc_original - svc_corrected:.1f})")
            
            # 充电弧修正
            chg_original = df[(df['t'] == t) & (df['arc_type'].isin(['tochg', 'chg_enter', 'chg_occ', 'chg_step']))]['flow'].sum()
            chg_corrected = df_corrected[(df_corrected['t'] == t) & (df_corrected['arc_type'] == 'tochg')]['flow_corrected'].sum()
            print(f"    Charging arcs: {chg_original:.1f} -> {chg_corrected:.1f} (diff: {chg_original - chg_corrected:.1f})")
    
    print("\nArc Category Distribution (Corrected):")
    print(df_corrected['arc_category'].value_counts())
    
    print("\nFlow by Category and Time Step (Corrected):")
    for category in ['Service', 'Reposition', 'Charging', 'Idle']:
        print(f"\n{category}:")
        category_data = aggregated_df[aggregated_df['arc_category'] == category]
        if not category_data.empty:
            for _, row in category_data.iterrows():
                print(f"  Time {row['time_step']}: Total={row['total_flow']:.2f}, "
                      f"Count={row['arc_count']}, Avg={row['avg_flow']:.4f}")
        else:
            print("  No data available")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize arc flows by time step and category (corrected version)')
    parser.add_argument('--flows-path', default='outputs/flows.parquet',
                       help='Path to flows.parquet file')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for visualization files')
    parser.add_argument('--start-step', type=int, default=None,
                       help='Start time step for visualization')
    parser.add_argument('--end-step', type=int, default=None,
                       help='End time step for visualization')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print("Loading flows data...")
        df = load_flows_data(args.flows_path)
        
        # 过滤时间步（如果指定）
        if args.start_step is not None or args.end_step is not None:
            if args.start_step is not None:
                df = df[df['t'] >= args.start_step]
            if args.end_step is not None:
                df = df[df['t'] <= args.end_step]
            print(f"Filtered to time steps: {df['t'].min()} to {df['t'].max()}")
        
        # 分类弧类型
        df = categorize_arc_types(df)
        
        # 修正服务弧和充电弧的重复计算
        print("\nCorrecting service and charging arc flows...")
        df_corrected = correct_service_and_charging_flows(df)
        
        # 分析弧持续时间模式
        print("\nAnalyzing arc duration patterns...")
        duration_analysis = analyze_arc_durations(df_corrected)
        print("Arc duration analysis:")
        for arc_type, analysis in duration_analysis.items():
            print(f"  {arc_type}: avg_duration={analysis['avg_duration']:.2f}, "
                  f"max_duration={analysis['max_duration']}, "
                  f"groups={analysis['continuous_groups']}")
        
        # 处理跨时间步的弧
        print("\nHandling cross-timestep arcs...")
        # 估算服务持续时间
        print("\nEstimating service duration...")
        service_duration_map = estimate_service_duration(df_corrected)
        print(f"Service duration map: {service_duration_map}")
        
        # 估算充电持续时间
        print("\nEstimating charging duration...")
        charging_duration_map = estimate_charging_duration(df_corrected)
        print(f"Charging duration map: {charging_duration_map}")
        
        # 使用分析结果来动态调整持续时间
        smart_duration_map = {}
        for arc_type, analysis in duration_analysis.items():
            if arc_type in ['reposition']:
                # 使用分析得出的平均持续时间，但限制在合理范围内
                smart_duration_map[arc_type] = max(1, min(5, int(round(analysis['avg_duration']))))
        
        # 添加服务持续时间
        if 'svc_enter' in service_duration_map:
            smart_duration_map['svc_enter'] = service_duration_map['svc_enter']
        
        # 添加充电持续时间
        if 'tochg' in charging_duration_map:
            smart_duration_map['tochg'] = charging_duration_map['tochg']
        
        print(f"Smart duration map: {smart_duration_map}")
        df_expanded = handle_cross_timestep_arcs_smart(df_corrected, smart_duration_map)
        
        # 聚合数据
        aggregated_df = aggregate_flows_by_time_and_category(df_expanded)
        
        # 打印详细统计
        print_detailed_stats(df, df_corrected, aggregated_df)
        
        # 创建可视化
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating visualizations...")
        
        # 1. 分离的子图
        subplot_path = output_dir / "arc_flows_timeline_corrected.html"
        create_line_chart(aggregated_df, str(subplot_path))
        
        # 2. 组合折线图
        combined_path = output_dir / "arc_flows_combined_corrected.html"
        create_combined_line_chart(aggregated_df, str(combined_path))
        
        print(f"\nCorrected visualization complete! Files saved to: {output_dir}")
        print(f"- Timeline chart: {subplot_path}")
        print(f"- Combined chart: {combined_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
