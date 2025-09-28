#!/usr/bin/env python3
"""
缓存清理工具
用于强制重新生成中间数据文件
"""

import os
import shutil
from pathlib import Path

def clear_grid_cache():
    """清理网格相关的缓存文件"""
    cache_dir = Path("data/intermediate")
    grid_files = [
        "nodes.parquet",
        "initial_inventory.parquet", 
        "time_index.json",
        "soc_index.json",
        "zone_index.json",
        "node_indexer.pkl",
        "grid_summary.json"
    ]
    
    print("清理网格缓存文件...")
    for file in grid_files:
        file_path = cache_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"  删除: {file}")
        else:
            print(f"  不存在: {file}")

def clear_reachability_cache():
    """清理可达性相关的缓存文件"""
    cache_dir = Path("data/intermediate")
    reachability_files = [
        "zone_station_best.parquet",
        "nearest_stations.json",
        "nearest_stations_pruning.csv",
        "reachability.parquet",
        "reachability.csv",
        "zone_station_energy_kept.csv",
        "zone_energy_nearest_agg.csv"
    ]
    
    print("\n清理可达性缓存文件...")
    for file in reachability_files:
        file_path = cache_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"  删除: {file}")
        else:
            print(f"  不存在: {file}")

def clear_solver_graph_cache():
    """清理求解器图缓存"""
    solver_graph_dir = Path("data/solver_graph")
    if solver_graph_dir.exists():
        print(f"\n清理求解器图缓存: {solver_graph_dir}")
        shutil.rmtree(solver_graph_dir)
    else:
        print(f"\n求解器图缓存不存在: {solver_graph_dir}")

def clear_all_cache():
    """清理所有缓存"""
    clear_grid_cache()
    clear_reachability_cache()
    clear_solver_graph_cache()
    print("\n✅ 所有缓存已清理完成！")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "grid":
            clear_grid_cache()
        elif action == "reachability":
            clear_reachability_cache()
        elif action == "solver":
            clear_solver_graph_cache()
        elif action == "all":
            clear_all_cache()
        else:
            print("用法: python clear_cache.py [grid|reachability|solver|all]")
    else:
        print("缓存清理工具")
        print("用法: python clear_cache.py [grid|reachability|solver|all]")
        print("  grid        - 清理网格缓存（包括time_index.json等）")
        print("  reachability - 清理可达性缓存")
        print("  solver      - 清理求解器图缓存")
        print("  all         - 清理所有缓存")
