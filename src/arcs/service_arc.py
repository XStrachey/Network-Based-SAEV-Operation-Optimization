# service_arc.py
# Service弧的具体实现
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# 确保src目录在Python路径中
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from arcs.arc_base import ArcBase, ArcCost, ArcMetadata, ArcFactory
from config.network_config import get_network_config as get_config
from utils.grid_utils import load_indexer
from utils.data_loader import load_od_matrix, load_base_ij


class ServiceArc(ArcBase):
    """Service弧实现 - 纯网络化的服务弧（三段式：enter, gate, exit）"""
    
    @property
    def arc_type_name(self) -> str:
        return "service"
    
    def generate_arcs(self, 
                     reachable_set: Set[Tuple[int, int, int]],
                     t0: Optional[int] = None,
                     t_hi: Optional[int] = None,
                     B: Optional[int] = None) -> List[ArcMetadata]:
        """
        生成service弧：
        - svc_enter: (i,t,l) -> (svc_in[i,j,t])，tau=0
        - svc_gate:  (svc_in[i,j,t]) -> (svc_out[i,j,t])，唯一容量弧（cap_hint = demand，成本稍后设为 -reward）
        - svc_exit:  (svc_out[i,j,t]) -> (j,t+tau,l-de)（带时间推进与 SOC 下降）
        """
        # 加载OD需求数据
        od = load_od_matrix(self.cfg)
        od = od[od["demand"] > 0].copy()
        
        # 时间过滤 - 修复：允许生成所有时间段的service弧
        if t0 is not None:  # 窗口模式
            # 允许生成所有时间段的service弧，让优化器决定可行性
            od = od[(od["t"] >= t0) & (od["t"] < t_hi)]
        
        if od.empty:
            return []
        
        # 加载距离信息
        bij = load_base_ij(self.cfg)
        srv = od.merge(bij, on=["i", "j"], how="inner")
        
        # 计算到达时间上限
        arrival_end = min(t_hi + int(B) if (t_hi is not None and B is not None) else self.gi.times[-1], self.gi.times[-1])
        soc_levels = np.array(self.gi.socs, dtype=int)
        
        arcs = []
        for _, r in srv.iterrows():
            i, j, t = int(r["i"]), int(r["j"]), int(r["t"])
            D_ijt = float(r["demand"])
            dist_km = float(r.get("dist_km", 0.0))
            
            # 基于距离和速度计算耗时
            tau = self._compute_travel_time(dist_km)
            t2 = t + tau
            
            if t2 > arrival_end or tau <= 0:
                continue
            
            # 计算能耗
            de = self._compute_energy_consumption(dist_km, t, tau, "de_per_km_srv")
            feas_soc = soc_levels[soc_levels >= de]
            
            if len(feas_soc) == 0:
                continue
            
            # 构造唯一的闸门节点
            in_id = self._pseudo_node_id("svc_in", i, j, t)
            out_id = self._pseudo_node_id("svc_out", i, j, t)
            req_key = f"{i}-{j}-{t}"
            
            # gate弧（单条容量弧，tau=0; cap_hint = D_ijt）
            gate_arc = ArcMetadata(
                arc_type="svc_gate",
                from_node_id=in_id,
                to_node_id=out_id,
                i=i,
                j=j,
                t=t,
                tau=0,
                de=0,
                req_key=req_key,
                cap_hint=D_ijt,
                dist_km=dist_km
            )
            arcs.append(gate_arc)
            
            # enter/exit弧（按可行SOC生成多条）
            for l in feas_soc:
                if not self._check_reachability(i, l, t, reachable_set):
                    continue
                if not self._check_reachability(j, l - de, t2, reachable_set):
                    continue
                
                from_id = self.gi.id_of(i, t, l)
                to_id = self.gi.id_of(j, t2, l - de)
                
                # enter弧：(grid) -> (svc_in)
                enter_arc = ArcMetadata(
                    arc_type="svc_enter",
                    from_node_id=from_id,
                    to_node_id=in_id,
                    i=i,
                    j=j,
                    t=t,
                    l=l,
                    tau=0,
                    de=0,
                    req_key=req_key,
                    dist_km=dist_km
                )
                arcs.append(enter_arc)
                
                # exit弧：(svc_out) -> (grid arrival)
                exit_arc = ArcMetadata(
                    arc_type="svc_exit",
                    from_node_id=out_id,
                    to_node_id=to_id,
                    i=i,
                    j=j,
                    t=t,
                    l=l,
                    tau=tau,
                    de=de,
                    req_key=req_key,
                    dist_km=dist_km
                )
                arcs.append(exit_arc)
        
        return arcs
    
    # 注意：compute_costs方法已移除，成本计算统一由ArcAssembly.compute_costs_batch处理


# 注册到工厂
ArcFactory.register_arc_type("service", ServiceArc)


def main():
    """测试service弧生成"""
    cfg = get_config()
    gi = load_indexer()
    
    # 加载可达性
    from arcs.arc_base import load_reachability_with_time
    reachable_set = load_reachability_with_time()
    
    # 创建service弧生成器
    service_generator = ServiceArc(cfg, gi)
    
    # 生成弧
    arcs = service_generator.generate_arcs(reachable_set)
    
    print(f"生成了 {len(arcs)} 条service弧")
    
    # 转换为DataFrame并保存
    df = service_generator.to_dataframe(arcs)
    
    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "service_arcs_new.parquet"
    df = service_generator.generate_and_save(reachable_set, output_path)
    
    print(f"保存到: {output_path}")
    print(f"弧数量: {len(df)}")
    
    if not df.empty:
        print("\n弧类型分布:")
        print(df['arc_type'].value_counts())
        
        print("\n前5条弧:")
        print(df.head())


if __name__ == "__main__":
    main()
