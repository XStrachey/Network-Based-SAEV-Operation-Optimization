#!/usr/bin/env python3
"""
分析 _08_build_solver_graph.py 生成的图规模和弧类型统计脚本

该脚本读取 solver_graph 目录下的数据文件，输出详细的图规模和弧类型分析报告。

使用方法:
    python analyze_solver_graph.py [--data-dir DIR] [--output-format FORMAT] [--save-report FILE]
    
参数:
    --data-dir: 数据目录路径 (默认: src/data/solver_graph)
    --output-format: 输出格式 (console/json/csv, 默认: console)
    --save-report: 保存报告到文件 (可选)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_solver_graph_data(data_dir: str = "src/data/solver_graph") -> Dict[str, Any]:
    """
    加载求解器图数据
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        包含元数据、节点和弧数据的字典
    """
    data_path = Path(data_dir)
    
    # 读取元数据
    meta_path = data_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {meta_path}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 读取节点数据
    nodes_path = data_path / "nodes.parquet"
    if not nodes_path.exists():
        raise FileNotFoundError(f"节点文件不存在: {nodes_path}")
    
    nodes_df = pd.read_parquet(nodes_path)
    
    # 读取弧数据
    arcs_path = data_path / "arcs.parquet"
    if not arcs_path.exists():
        raise FileNotFoundError(f"弧文件不存在: {arcs_path}")
    
    arcs_df = pd.read_parquet(arcs_path)
    
    return {
        "meta": meta,
        "nodes": nodes_df,
        "arcs": arcs_df
    }


def analyze_graph_scale(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析图规模信息
    
    Args:
        data: 包含元数据、节点和弧数据的字典
        
    Returns:
        图规模分析结果
    """
    meta = data["meta"]
    nodes_df = data["nodes"]
    arcs_df = data["arcs"]
    
    # 基本规模信息
    scale_info = {
        "basic": {
            "nodes_total": len(nodes_df),
            "arcs_total": len(arcs_df),
            "time_window": {
                "t0": meta.get("t0"),
                "H": meta.get("H"),
                "t_hi": meta.get("t_hi")
            },
            "supply_total": meta.get("sup_total", 0.0)
        }
    }
    
    # 节点分析
    nodes_with_supply = (nodes_df["supply"] > 0).sum()
    nodes_with_negative_supply = (nodes_df["supply"] < 0).sum()
    
    scale_info["nodes"] = {
        "total": len(nodes_df),
        "with_positive_supply": nodes_with_supply,
        "with_negative_supply": nodes_with_negative_supply,
        "with_zero_supply": len(nodes_df) - nodes_with_supply - nodes_with_negative_supply,
        "time_range": {
            "min": float(nodes_df["t"].min()) if "t" in nodes_df.columns else None,
            "max": float(nodes_df["t"].max()) if "t" in nodes_df.columns else None
        },
        "soc_range": {
            "min": float(nodes_df["soc"].min()) if "soc" in nodes_df.columns else None,
            "max": float(nodes_df["soc"].max()) if "soc" in nodes_df.columns else None
        },
        "zones_count": int(nodes_df["zone"].nunique()) if "zone" in nodes_df.columns else None
    }
    
    # 弧分析
    scale_info["arcs"] = {
        "total": len(arcs_df),
        "unique_arc_types": int(arcs_df["arc_type"].nunique()) if "arc_type" in arcs_df.columns else 0,
        "cost_range": {
            "min": float(arcs_df["cost"].min()) if "cost" in arcs_df.columns else None,
            "max": float(arcs_df["cost"].max()) if "cost" in arcs_df.columns else None,
            "mean": float(arcs_df["cost"].mean()) if "cost" in arcs_df.columns else None
        },
        "capacity_range": {
            "min": float(arcs_df["capacity"].min()) if "capacity" in arcs_df.columns else None,
            "max": float(arcs_df["capacity"].max()) if "capacity" in arcs_df.columns else None,
            "mean": float(arcs_df["capacity"].mean()) if "capacity" in arcs_df.columns else None
        }
    }
    
    return scale_info


def analyze_arc_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析弧类型分布
    
    Args:
        data: 包含元数据、节点和弧数据的字典
        
    Returns:
        弧类型分析结果
    """
    arcs_df = data["arcs"]
    
    if "arc_type" not in arcs_df.columns:
        return {"error": "弧数据中缺少 arc_type 列"}
    
    # 弧类型统计
    arc_type_counts = arcs_df["arc_type"].value_counts().sort_index()
    
    # 计算百分比
    total_arcs = len(arcs_df)
    arc_type_percentages = (arc_type_counts / total_arcs * 100).round(2)
    
    # 弧类型详细分析
    arc_type_details = {}
    for arc_type in arc_type_counts.index:
        subset = arcs_df[arcs_df["arc_type"] == arc_type]
        
        details = {
            "count": int(arc_type_counts[arc_type]),
            "percentage": float(arc_type_percentages[arc_type]),
            "cost_stats": {
                "min": float(subset["cost"].min()) if "cost" in subset.columns else None,
                "max": float(subset["cost"].max()) if "cost" in subset.columns else None,
                "mean": float(subset["cost"].mean()) if "cost" in subset.columns else None
            },
            "capacity_stats": {
                "min": float(subset["capacity"].min()) if "capacity" in subset.columns else None,
                "max": float(subset["capacity"].max()) if "capacity" in subset.columns else None,
                "mean": float(subset["capacity"].mean()) if "capacity" in subset.columns else None
            }
        }
        
        # 如果有时间信息，分析时间分布
        if "t" in subset.columns:
            details["time_stats"] = {
                "min": float(subset["t"].min()),
                "max": float(subset["t"].max()),
                "unique_times": int(subset["t"].nunique())
            }
        
        arc_type_details[arc_type] = details
    
    return {
        "summary": {
            "total_arc_types": len(arc_type_counts),
            "dominant_type": arc_type_counts.index[arc_type_counts.argmax()],
            "dominant_count": int(arc_type_counts.max()),
            "dominant_percentage": float(arc_type_percentages.max())
        },
        "counts": arc_type_counts.to_dict(),
        "percentages": arc_type_percentages.to_dict(),
        "details": arc_type_details
    }


def print_analysis_report(scale_info: Dict[str, Any], arc_types_info: Dict[str, Any], meta: Dict[str, Any]):
    """
    打印分析报告
    
    Args:
        scale_info: 图规模分析结果
        arc_types_info: 弧类型分析结果
        meta: 元数据
    """
    print("=" * 80)
    print("求解器图规模与弧类型分析报告")
    print("=" * 80)
    
    # 基本信息
    print("\n📊 基本信息")
    print("-" * 40)
    basic = scale_info["basic"]
    print(f"时间窗口: t0={basic['time_window']['t0']}, H={basic['time_window']['H']}, t_hi={basic['time_window']['t_hi']}")
    print(f"节点总数: {basic['nodes_total']:,}")
    print(f"弧总数: {basic['arcs_total']:,}")
    print(f"供给总量: {basic['supply_total']}")
    
    # 节点详细信息
    print("\n🔢 节点详细信息")
    print("-" * 40)
    nodes_info = scale_info["nodes"]
    print(f"总节点数: {nodes_info['total']:,}")
    print(f"正供给节点数: {nodes_info['with_positive_supply']:,}")
    print(f"负供给节点数: {nodes_info['with_negative_supply']:,}")
    print(f"零供给节点数: {nodes_info['with_zero_supply']:,}")
    
    if nodes_info["time_range"]["min"] is not None:
        print(f"时间范围: {nodes_info['time_range']['min']:.0f} - {nodes_info['time_range']['max']:.0f}")
    if nodes_info["soc_range"]["min"] is not None:
        print(f"SOC范围: {nodes_info['soc_range']['min']:.0f} - {nodes_info['soc_range']['max']:.0f}")
    if nodes_info["zones_count"] is not None:
        print(f"区域数量: {nodes_info['zones_count']}")
    
    # 弧详细信息
    print("\n🔗 弧详细信息")
    print("-" * 40)
    arcs_info = scale_info["arcs"]
    print(f"总弧数: {arcs_info['total']:,}")
    print(f"弧类型数: {arcs_info['unique_arc_types']}")
    
    if arcs_info["cost_range"]["min"] is not None:
        print(f"成本范围: {arcs_info['cost_range']['min']:.4f} - {arcs_info['cost_range']['max']:.4f}")
        print(f"平均成本: {arcs_info['cost_range']['mean']:.4f}")
    
    if arcs_info["capacity_range"]["min"] is not None:
        print(f"容量范围: {arcs_info['capacity_range']['min']:.0f} - {arcs_info['capacity_range']['max']:.0f}")
        print(f"平均容量: {arcs_info['capacity_range']['mean']:.0f}")
    
    # 弧类型分布
    print("\n📈 弧类型分布")
    print("-" * 40)
    
    if "error" in arc_types_info:
        print(f"错误: {arc_types_info['error']}")
        return
    
    summary = arc_types_info["summary"]
    print(f"弧类型总数: {summary['total_arc_types']}")
    print(f"主导类型: {summary['dominant_type']} ({summary['dominant_count']:,} 条, {summary['dominant_percentage']:.1f}%)")
    
    print("\n详细分布:")
    print(f"{'弧类型':<15} {'数量':<12} {'百分比':<8} {'平均成本':<12} {'平均容量':<12}")
    print("-" * 65)
    
    for arc_type, details in arc_types_info["details"].items():
        cost_mean = details["cost_stats"]["mean"]
        capacity_mean = details["capacity_stats"]["mean"]
        
        cost_str = f"{cost_mean:.4f}" if cost_mean is not None else "N/A"
        capacity_str = f"{capacity_mean:.0f}" if capacity_mean is not None else "N/A"
        
        print(f"{arc_type:<15} {details['count']:<12,} {details['percentage']:<8.1f}% {cost_str:<12} {capacity_str:<12}")
    
    # 特殊节点信息
    print("\n🎯 特殊节点信息")
    print("-" * 40)
    if meta.get("sink_node_id") is not None:
        print(f"超级汇点节点ID: {meta['sink_node_id']}")
    
    print("\n" + "=" * 80)


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def save_json_report(scale_info: Dict[str, Any], arc_types_info: Dict[str, Any], 
                     meta: Dict[str, Any], output_file: str):
    """保存JSON格式报告"""
    report = {
        "metadata": convert_numpy_types(meta),
        "scale_analysis": convert_numpy_types(scale_info),
        "arc_types_analysis": convert_numpy_types(arc_types_info),
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"报告已保存到: {output_file}")


def save_csv_report(scale_info: Dict[str, Any], arc_types_info: Dict[str, Any], 
                    meta: Dict[str, Any], output_file: str):
    """保存CSV格式报告"""
    # 创建弧类型统计表
    arc_types_data = []
    for arc_type, details in arc_types_info["details"].items():
        arc_types_data.append({
            "arc_type": arc_type,
            "count": details["count"],
            "percentage": details["percentage"],
            "avg_cost": details["cost_stats"]["mean"],
            "avg_capacity": details["capacity_stats"]["mean"]
        })
    
    arc_types_df = pd.DataFrame(arc_types_data)
    
    # 创建基本信息表
    basic_info = pd.DataFrame([{
        "metric": "total_nodes",
        "value": scale_info["basic"]["nodes_total"]
    }, {
        "metric": "total_arcs", 
        "value": scale_info["basic"]["arcs_total"]
    }, {
        "metric": "total_supply",
        "value": scale_info["basic"]["supply_total"]
    }, {
        "metric": "time_window_t0",
        "value": scale_info["basic"]["time_window"]["t0"]
    }, {
        "metric": "time_window_H",
        "value": scale_info["basic"]["time_window"]["H"]
    }, {
        "metric": "time_window_t_hi",
        "value": scale_info["basic"]["time_window"]["t_hi"]
    }])
    
    # 保存到CSV文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        arc_types_df.to_excel(writer, sheet_name='Arc_Types', index=False)
        basic_info.to_excel(writer, sheet_name='Basic_Info', index=False)
    
    print(f"报告已保存到: {output_file}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="分析 _08_build_solver_graph.py 生成的图规模和弧类型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyze_solver_graph.py
  python analyze_solver_graph.py --data-dir custom/path
  python analyze_solver_graph.py --output-format json --save-report report.json
  python analyze_solver_graph.py --output-format csv --save-report report.xlsx
        """
    )
    
    parser.add_argument(
        "--data-dir",
        default="src/data/solver_graph",
        help="数据目录路径 (默认: src/data/solver_graph)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "csv"],
        default="console",
        help="输出格式 (默认: console)"
    )
    
    parser.add_argument(
        "--save-report",
        help="保存报告到文件 (JSON或Excel格式)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只输出错误信息"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 加载数据
        if not args.quiet:
            print("正在加载求解器图数据...")
        data = load_solver_graph_data(args.data_dir)
        
        # 分析图规模
        if not args.quiet:
            print("正在分析图规模...")
        scale_info = analyze_graph_scale(data)
        
        # 分析弧类型
        if not args.quiet:
            print("正在分析弧类型...")
        arc_types_info = analyze_arc_types(data)
        
        # 根据输出格式处理结果
        if args.output_format == "console":
            print_analysis_report(scale_info, arc_types_info, data["meta"])
        elif args.output_format == "json":
            if args.save_report:
                save_json_report(scale_info, arc_types_info, data["meta"], args.save_report)
            else:
                report = {
                    "metadata": data["meta"],
                    "scale_analysis": scale_info,
                    "arc_types_analysis": arc_types_info
                }
                print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.output_format == "csv":
            if args.save_report:
                save_csv_report(scale_info, arc_types_info, data["meta"], args.save_report)
            else:
                print("CSV格式需要指定输出文件，请使用 --save-report 参数")
                return 1
        
        # 保存报告（如果指定）
        if args.save_report and args.output_format == "console":
            # 默认保存JSON格式
            save_json_report(scale_info, arc_types_info, data["meta"], args.save_report)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
