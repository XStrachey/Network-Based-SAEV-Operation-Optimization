# idle_arc.py
# Idle弧的具体实现
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

# 确保src目录在Python路径中
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from arcs.arc_base import ArcBase, ArcCost, ArcMetadata, ArcFactory
from config.network_config import get_network_config as get_config
from utils.grid_utils import load_indexer


class IdleArc(ArcBase):
    """Idle弧实现 - 车辆在同一区域等待一个时间步"""
    
    @property
    def arc_type_name(self) -> str:
        return "idle"
    
    def generate_arcs(self, 
                     reachable_set: Set[Tuple[int, int, int]],
                     t0: Optional[int] = None,
                     t_hi: Optional[int] = None,
                     B: Optional[int] = None) -> List[ArcMetadata]:
        """
        生成idle弧：(i,t,l) -> (i,t+1,l)
        
        Args:
            reachable_set: 可达的(zone, soc, time)集合
            t0: 窗口起始时间
            t_hi: 窗口结束时间
            B: Halo步数
        """
        # 加载节点数据
        nodes_df = pd.read_parquet("data/intermediate/nodes.parquet")
        
        # 确定时间范围
        if t0 is not None:  # 窗口模式
            max_dep_t = min(t_hi - 1, self.gi.times[-1] - 1)
            nodes_df = nodes_df[(nodes_df["t"] >= t0) & (nodes_df["t"] <= max_dep_t)]
        else:  # 全生成模式
            max_t = self.gi.times[-1]
            nodes_df = nodes_df[nodes_df["t"] + 1 <= max_t]
        
        if nodes_df.empty:
            return []
        
        arcs = []
        for _, row in nodes_df.iterrows():
            zone = int(row["zone"])
            time = int(row["t"])
            soc = int(row["soc"])
            from_node_id = int(row["node_id"])
            
            # 检查起始节点是否可达
            if not self._check_reachability(zone, soc, time, reachable_set):
                continue
            
            # 计算目标节点
            to_time = time + 1
            to_node_id = self.gi.id_of(zone, to_time, soc)
            
            # 检查目标节点是否可达
            if not self._check_reachability(zone, soc, to_time, reachable_set):
                continue
            
            # 创建弧元数据
            arc = ArcMetadata(
                arc_type="idle",
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                i=zone,
                t=time,
                l=soc,
                tau=1,
                de=0,  # idle不消耗能量
                dist_km=0.0  # idle不移动
            )
            
            arcs.append(arc)
        
        return arcs
    
    # 注意：compute_costs方法已移除，成本计算统一由ArcAssembly.compute_costs_batch处理


# 注册到工厂
ArcFactory.register_arc_type("idle", IdleArc)


def main():
    """测试idle弧生成"""
    cfg = get_config()
    gi = load_indexer()
    
    # 加载可达性
    from arcs.arc_base import load_reachability_with_time
    reachable_set = load_reachability_with_time()
    
    # 创建idle弧生成器
    idle_generator = IdleArc(cfg, gi)
    
    # 生成弧
    arcs = idle_generator.generate_arcs(reachable_set)
    
    print(f"生成了 {len(arcs)} 条idle弧")
    
    # 转换为DataFrame并保存
    df = idle_generator.to_dataframe(arcs)
    
    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "idle_arcs_new.parquet"
    df = idle_generator.generate_and_save(reachable_set, output_path)
    
    print(f"保存到: {output_path}")
    print(f"弧数量: {len(df)}")
    
    if not df.empty:
        print("\n前5条弧:")
        print(df.head())
        print(f"\n弧类型分布:")
        print(df['arc_type'].value_counts())


if __name__ == "__main__":
    main()
