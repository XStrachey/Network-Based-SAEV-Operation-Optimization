#!/usr/bin/env python3
"""
基于JSON配置的场景运行脚本
支持通过不同的JSON配置文件运行不同的定价策略场景

使用方法:
    python run_scenario.py --scenario fcfs
    python run_scenario.py --scenario tou_peak
    python run_scenario.py --scenario tou_off_peak
    python run_scenario.py --list  # 列出所有可用场景
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

def list_available_scenarios():
    """列出所有可用的场景配置"""
    # 从src/pipeline目录运行时，需要回到项目根目录
    config_dir = Path("../../configs")
    if not config_dir.exists():
        print("❌ configs目录不存在")
        return
    
    json_files = list(config_dir.glob("*.json"))
    if not json_files:
        print("❌ 没有找到任何场景配置文件")
        return
    
    print("📋 可用场景:")
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            name = config.get('name', json_file.stem)
            description = config.get('description', '无描述')
            print(f"  - {json_file.stem}: {name}")
            print(f"    描述: {description}")
        except Exception as e:
            print(f"  - {json_file.stem}: 配置文件错误 - {e}")
        print()

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 成功!")
        if result.stdout:
            print("输出:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return False

def load_scenario_config(scenario: str):
    """加载场景配置"""
    # 从src/pipeline目录运行时，需要回到项目根目录
    config_path = f"../../configs/{scenario}.json"
    if not Path(config_path).exists():
        print(f"❌ 场景配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def main():
    parser = argparse.ArgumentParser(description='基于JSON配置的场景运行脚本')
    parser.add_argument('--scenario', '-s', help='场景名称 (如: fcfs, tou_peak)')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用场景')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_scenarios()
        return
    
    if not args.scenario:
        print("❌ 请指定场景名称，使用 --scenario 参数")
        print("💡 使用 --list 查看所有可用场景")
        sys.exit(1)
    
    print(f"🚀 开始运行场景: {args.scenario}")
    print(f"📁 工作目录: {Path.cwd()}")
    
    # 加载场景配置
    config = load_scenario_config(args.scenario)
    if config is None:
        sys.exit(1)
    
    print(f"📊 场景: {config.get('name', args.scenario)}")
    print(f"📝 描述: {config.get('description', '无描述')}")
    
    # 检查虚拟环境（从src/pipeline目录运行时，需要回到项目根目录）
    venv_path = Path("../../venv")
    if not venv_path.exists():
        print("❌ 虚拟环境不存在，请先运行: python -m venv venv")
        sys.exit(1)
    
    # 激活虚拟环境的命令前缀
    if sys.platform == "win32":
        python_cmd = ["../../venv\\Scripts\\python.exe"]
    else:
        python_cmd = ["../../venv/bin/python"]
    
    # 获取输出目录
    output_dir = config.get('output', {}).get('dir', '../../outputs')
    
    # 为每个场景创建独立的中间文件目录（放在outputs目录下）
    solver_graph_dir = f"{output_dir}/solver_graph"
    
    # 步骤1: 构建求解图
    cmd1 = python_cmd + ["_01_build_solver_graph.py", "--scenario", args.scenario, "--out", solver_graph_dir]
    if not run_command(cmd1, "构建求解图"):
        sys.exit(1)
    
    # 步骤2: 求解最小费用流
    cmd2 = python_cmd + ["_02_solve_graph_mincost.py", 
                         "--nodes", f"{solver_graph_dir}/nodes.parquet",
                         "--arcs", f"{solver_graph_dir}/arcs.parquet", 
                         "--meta", f"{solver_graph_dir}/meta.json",
                         "--out", output_dir]
    if not run_command(cmd2, "求解最小费用流"):
        sys.exit(1)
    
    # 步骤3: 可视化结果
    cmd3 = python_cmd + ["visualize_arc_flows_corrected.py", "--flows-path", f"{output_dir}/flows.parquet", "--output-dir", f"{output_dir}/viz"]
    if not run_command(cmd3, "生成可视化结果"):
        sys.exit(1)
    
    # 步骤4: 导出简化图结构
    cmd4 = python_cmd + ["../../src/analysis/export_simple_graph.py", 
                         "--data-dir", solver_graph_dir,
                         "--flows-dir", output_dir,
                         "--output", f"{output_dir}/simple_graph.json"]
    if not run_command(cmd4, "导出简化图结构"):
        sys.exit(1)
    
    # 步骤5: 生成图可视化
    cmd5 = python_cmd + ["../../src/viz/viz_graph.py", 
                         "--input", f"{output_dir}/simple_graph.json",
                         "--output", f"{output_dir}/graph.html"]
    if not run_command(cmd5, "生成图可视化"):
        sys.exit(1)
    
    print(f"\n🎉 场景 {args.scenario} 运行完成!")
    print(f"📈 结果文件:")
    print(f"  - 流量数据: {output_dir}/flows.parquet")
    print(f"  - 求解摘要: {output_dir}/solve_summary.json")
    print(f"  - 流量可视化: {output_dir}/viz/arc_flows_*.html")
    print(f"  - 简化图结构: {output_dir}/simple_graph.json")
    print(f"  - 图可视化: {output_dir}/graph.html")
    
    # 显示求解摘要
    summary_file = Path(f"{output_dir}/solve_summary.json")
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\n📊 求解结果摘要:")
        print(f"  - 状态: {summary['status']}")
        print(f"  - 目标值: {summary['objective']}")
        print(f"  - 总流量: {summary['total_flow']}")
        print(f"  - 节点数: {summary['nodes']}")
        print(f"  - 弧数: {summary['arcs']}")

if __name__ == "__main__":
    main()
