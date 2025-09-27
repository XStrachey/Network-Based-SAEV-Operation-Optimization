#!/usr/bin/env python3
"""
弧流量可视化脚本（修正版）
用于正确显示每个时间段五种类型弧流量的折线图
修正了服务弧三段式结构的重复计算问题
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

def load_flows_data(flows_path: str = "src/outputs/flows.parquet") -> pd.DataFrame:
    """加载流量数据"""
    flows_file = Path(flows_path)
    if not flows_file.exists():
        raise FileNotFoundError(f"Flows file not found: {flows_path}")
    
    df = pd.read_parquet(flows_file)
    print(f"Loaded {len(df)} flow records")
    print(f"Time range: {df['t'].min()} to {df['t'].max()}")
    print(f"Arc types: {df['arc_type'].unique()}")
    
    return df

def categorize_arc_types(df: pd.DataFrame) -> pd.DataFrame:
    """将弧类型分类为五种主要类型"""
    def categorize_arc(arc_type):
        if arc_type in ['svc_enter', 'svc_gate', 'svc_exit']:
            return 'Service'
        elif arc_type in ['chg_enter', 'chg_occ', 'chg_step']:
            return 'Charging'
        elif arc_type == 'reposition':
            return 'Reposition'
        elif arc_type == 'idle':
            return 'Idle'
        elif arc_type == 'tochg':
            return 'ToCharging'
        else:
            return 'Other'
    
    df = df.copy()
    df['arc_category'] = df['arc_type'].apply(categorize_arc)
    
    return df

def correct_service_and_charging_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    修正服务弧和充电弧的重复计算问题
    
    服务弧采用三段式结构：svc_enter -> svc_gate -> svc_exit
    充电弧采用四段式结构：tochg -> chg_enter -> chg_occ -> chg_step
    
    同一辆车会被重复计算，需要修正为实际的车辆数：
    - 服务弧：只计算svc_gate的流量（实际容量约束）
    - 充电弧：只计算tochg的流量（最能正确对应车辆实际数量）
    """
    df_corrected = df.copy()
    
    # 对于服务弧，只计算svc_gate的流量（因为它是实际的容量约束）
    service_enter_exit_mask = df_corrected['arc_type'].isin(['svc_enter', 'svc_exit'])
    df_corrected.loc[service_enter_exit_mask, 'flow_corrected'] = 0
    
    # 对于svc_gate，保持原始流量
    service_gate_mask = df_corrected['arc_type'] == 'svc_gate'
    df_corrected.loc[service_gate_mask, 'flow_corrected'] = df_corrected.loc[service_gate_mask, 'flow']
    
    # 对于充电弧，只计算tochg的流量（最能正确对应车辆实际数量）
    charging_other_mask = df_corrected['arc_type'].isin(['chg_enter', 'chg_occ', 'chg_step'])
    df_corrected.loc[charging_other_mask, 'flow_corrected'] = 0
    
    # 对于tochg，保持原始流量
    charging_tochg_mask = df_corrected['arc_type'] == 'tochg'
    df_corrected.loc[charging_tochg_mask, 'flow_corrected'] = df_corrected.loc[charging_tochg_mask, 'flow']
    
    # 对于其他弧类型（idle, reposition等），保持原始流量
    other_mask = ~df_corrected['arc_type'].isin([
        'svc_enter', 'svc_gate', 'svc_exit', 
        'tochg', 'chg_enter', 'chg_occ', 'chg_step'
    ])
    df_corrected.loc[other_mask, 'flow_corrected'] = df_corrected.loc[other_mask, 'flow']
    
    return df_corrected

def aggregate_flows_by_time_and_category(df: pd.DataFrame) -> pd.DataFrame:
    """按时间段和弧类别聚合流量（使用修正后的流量）"""
    # 按时间步和弧类别聚合
    aggregated = df.groupby(['t', 'arc_category'])['flow_corrected'].agg(['sum', 'count']).reset_index()
    aggregated.columns = ['time_step', 'arc_category', 'total_flow', 'arc_count']
    aggregated['avg_flow'] = aggregated['total_flow'] / aggregated['arc_count'].replace(0, 1)
    
    return aggregated

def create_line_chart(aggregated_df: pd.DataFrame, output_path: str = "arc_flows_timeline_corrected.html"):
    """创建五种类型弧流量的折线图"""
    main_categories = ['Service', 'Reposition', 'Charging', 'Idle', 'ToCharging']
    
    # 创建子图 (2行3列，因为现在有5个类别)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=main_categories,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 颜色映射
    colors = {
        'Service': '#1f77b4',
        'Reposition': '#ff7f0e', 
        'Charging': '#2ca02c',
        'Idle': '#d62728',
        'ToCharging': '#9467bd'
    }
    
    # 为每个类别创建图表
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
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
        'Idle': '#d62728',
        'ToCharging': '#9467bd'
    }
    
    fig = go.Figure()
    
    # 为每个类别添加折线
    for category in ['Service', 'Reposition', 'Charging', 'Idle', 'ToCharging']:
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
            svc_corrected = df_corrected[(df_corrected['t'] == t) & (df_corrected['arc_type'] == 'svc_gate')]['flow_corrected'].sum()
            print(f"    Service arcs: {svc_original:.1f} -> {svc_corrected:.1f} (diff: {svc_original - svc_corrected:.1f})")
            
            # 充电弧修正
            chg_original = df[(df['t'] == t) & (df['arc_type'].isin(['tochg', 'chg_enter', 'chg_occ', 'chg_step']))]['flow'].sum()
            chg_corrected = df_corrected[(df_corrected['t'] == t) & (df_corrected['arc_type'] == 'tochg')]['flow_corrected'].sum()
            print(f"    Charging arcs: {chg_original:.1f} -> {chg_corrected:.1f} (diff: {chg_original - chg_corrected:.1f})")
    
    print("\nArc Category Distribution (Corrected):")
    print(df_corrected['arc_category'].value_counts())
    
    print("\nFlow by Category and Time Step (Corrected):")
    for category in ['Service', 'Reposition', 'Charging', 'Idle', 'ToCharging']:
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
    parser.add_argument('--flows-path', default='src/outputs/flows.parquet',
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
        
        # 聚合数据
        aggregated_df = aggregate_flows_by_time_and_category(df_corrected)
        
        # 打印详细统计
        print_detailed_stats(df, df_corrected, aggregated_df)
        
        # 创建可视化
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
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
