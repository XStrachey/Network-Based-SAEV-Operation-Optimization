# reposition_arc.py
# Reposition弧的具体实现
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

import numpy as np
import pandas as pd

# 确保src目录在Python路径中
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from arcs.arc_base import ArcBase, ArcCost, ArcMetadata, ArcFactory
from config.network_config import get_network_config as get_config
from utils.grid_utils import load_indexer
from utils.data_loader import load_base_ij


class RepositionArc(ArcBase):
    """Reposition弧实现 - 需求驱动的重定位弧生成"""
    
    def __init__(self, cfg=None, gi: GridIndexers = None, inter_dir: str = "data/intermediate"):
        super().__init__(cfg, gi)
        self.inter_dir = inter_dir
    
    @property
    def arc_type_name(self) -> str:
        return "reposition"
    
    def _compute_reposition_demand(self, t0: Optional[int] = None, t_hi: Optional[int] = None) -> pd.DataFrame:
        """
        计算重定位需求矩阵 - 基于需求模式的重定位策略
        
        新的策略：
        1. 不依赖库存数据（因为库存是优化结果）
        2. 基于OD需求模式识别重定位机会
        3. 为高需求区域生成重定位弧
        """
        # 加载服务需求数据
        from utils.data_loader import load_od_matrix
        od = load_od_matrix(self.cfg)
        
        # 时间过滤
        if t0 is not None and t_hi is not None:
            od = od[(od["t"] >= t0) & (od["t"] <= t_hi - 1)]
        
        if od.empty:
            return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
        
        # 计算每个区域在每个时刻的需求模式
        zone_demand_pattern = self._analyze_demand_patterns(od)
        
        if zone_demand_pattern.empty:
            print("[reposition_arc] No demand patterns found")
            return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
        
        # 基于需求模式生成重定位需求
        reposition_demand = self._generate_reposition_from_patterns(zone_demand_pattern, od)
        
        if not reposition_demand:
            print("[reposition_arc] No reposition demand generated")
            return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
        
        demand_df = pd.DataFrame(reposition_demand)
        print(f"[reposition_arc] Generated {len(demand_df)} reposition demand records")
        
        # 去重和聚合
        demand_df = demand_df.groupby(['i', 'j', 't']).agg({
            'reposition_demand': 'sum'
        }).reset_index()
        
        print(f"[reposition_arc] After aggregation: {len(demand_df)} records")
        print(f"[reposition_arc] Total reposition demand: {demand_df['reposition_demand'].sum():.2f}")
        
        # 过滤掉重定位需求过小的OD对
        min_demand_threshold = getattr(self.cfg.pruning, 'min_reposition_demand', 0.1)
        print(f"[reposition_arc] Using min demand threshold: {min_demand_threshold}")
        
        before_filter = len(demand_df)
        demand_df = demand_df[demand_df['reposition_demand'] >= min_demand_threshold]
        after_filter = len(demand_df)
        
        print(f"[reposition_arc] After filtering: {before_filter} -> {after_filter} records")
        
        if not demand_df.empty:
            print("[reposition_arc] Final reposition demand:")
            print(demand_df.head(10))
        
        return demand_df
    
    def _analyze_demand_patterns(self, od: pd.DataFrame) -> pd.DataFrame:
        """
        分析需求模式，识别重定位机会
        
        Args:
            od: OD需求数据
            
        Returns:
            包含需求模式分析的DataFrame
        """
        print("[reposition_arc] Analyzing demand patterns...")
        
        # 计算每个区域在每个时刻的总需求（作为出发区域）
        zone_outbound_demand = od.groupby(['i', 't'])['demand'].sum().reset_index()
        zone_outbound_demand.columns = ['zone', 't', 'outbound_demand']
        
        # 计算每个区域在每个时刻的总需求（作为到达区域）
        zone_inbound_demand = od.groupby(['j', 't'])['demand'].sum().reset_index()
        zone_inbound_demand.columns = ['zone', 't', 'inbound_demand']
        
        # 合并出站和入站需求
        zone_demand = zone_outbound_demand.merge(zone_inbound_demand, on=['zone', 't'], how='outer')
        zone_demand['outbound_demand'] = zone_demand['outbound_demand'].fillna(0.0)
        zone_demand['inbound_demand'] = zone_demand['inbound_demand'].fillna(0.0)
        
        # 计算净需求（入站 - 出站）
        zone_demand['net_demand'] = zone_demand['inbound_demand'] - zone_demand['outbound_demand']
        
        # 计算需求强度（总需求）
        zone_demand['total_demand'] = zone_demand['inbound_demand'] + zone_demand['outbound_demand']
        
        print(f"[reposition_arc] Analyzed {len(zone_demand)} zone-time combinations")
        print(f"[reposition_arc] Net demand range: {zone_demand['net_demand'].min():.2f} to {zone_demand['net_demand'].max():.2f}")
        print(f"[reposition_arc] Total demand range: {zone_demand['total_demand'].min():.2f} to {zone_demand['total_demand'].max():.2f}")
        
        return zone_demand
    
    def _generate_reposition_from_patterns(self, zone_demand: pd.DataFrame, od: pd.DataFrame) -> List[Dict]:
        """
        基于需求模式生成重定位需求
        
        Args:
            zone_demand: 需求模式分析结果
            od: 原始OD数据
            
        Returns:
            重定位需求列表
        """
        print("[reposition_arc] Generating reposition demand from patterns...")
        
        reposition_demand = []
        
        # 获取所有区域
        all_zones = sorted(zone_demand['zone'].unique())
        
        # 重定位参数
        reposition_ratio = self.cfg.pruning.reposition_demand_ratio
        max_reposition_pairs = self.cfg.pruning.max_reposition_pairs_per_zone
        high_demand_threshold = self.cfg.pruning.high_demand_threshold  # 高需求阈值（基于实际数据：20和110）
        
        for _, row in zone_demand.iterrows():
            zone, t = int(row['zone']), int(row['t'])
            net_demand = row['net_demand']
            total_demand = row['total_demand']
            
            # 只考虑有足够需求的区域
            if total_demand < 0.1:
                continue
            
            # 策略1：从净需求为负的区域（供给过剩）到净需求为正的区域（需求过剩）
            if net_demand < -0.1:  # 供给过剩区域
                # 找到需求过剩的区域作为目标
                demand_surplus_zones = zone_demand[
                    (zone_demand['t'] == t) & 
                    (zone_demand['net_demand'] > 0.1) &
                    (zone_demand['zone'] != zone)
                ].copy()
                
                if not demand_surplus_zones.empty:
                    # 选择需求过剩最严重的几个目标区域
                    top_targets = demand_surplus_zones.nlargest(max_reposition_pairs, 'net_demand')
                    
                    for _, target in top_targets.iterrows():
                        target_zone = int(target['zone'])
                        
                        # 重定位需求基于供给过剩量和需求过剩量
                        reposition_val = min(abs(net_demand), target['net_demand']) * reposition_ratio
                        
                        if reposition_val > 0.1:
                            reposition_demand.append({
                                'i': zone,
                                'j': target_zone,
                                't': t,
                                'reposition_demand': reposition_val
                            })
            
            # 策略2：为高需求区域生成重定位弧（即使供需平衡）
            elif total_demand > high_demand_threshold:  # 高需求区域
                # 找到其他高需求区域作为重定位目标
                high_demand_zones = zone_demand[
                    (zone_demand['t'] == t) & 
                    (zone_demand['total_demand'] > high_demand_threshold) &
                    (zone_demand['zone'] != zone)
                ].copy()
                
                if not high_demand_zones.empty:
                    # 选择总需求最高的几个目标区域
                    top_targets = high_demand_zones.nlargest(max_reposition_pairs, 'total_demand')
                    
                    for _, target in top_targets.iterrows():
                        target_zone = int(target['zone'])
                        
                        # 重定位需求基于总需求
                        reposition_val = min(total_demand, target['total_demand']) * reposition_ratio * 0.5
                        
                        if reposition_val > 0.1:
                            reposition_demand.append({
                                'i': zone,
                                'j': target_zone,
                                't': t,
                                'reposition_demand': reposition_val
                            })
        
        print(f"[reposition_arc] Generated {len(reposition_demand)} reposition opportunities")
        return reposition_demand
    
    def _load_current_inventory(self, t0: Optional[int] = None, t_hi: Optional[int] = None) -> pd.DataFrame:
        """
        加载当期车辆库存数据
        
        Args:
            t0: 开始时间步
            t_hi: 结束时间步
            
        Returns:
            包含 zone, t, soc, supply 列的 DataFrame
        """
        try:
            # 尝试从solver_graph节点数据中加载
            nodes_path = Path("data/solver_graph/nodes.parquet")
            if nodes_path.exists():
                nodes_df = pd.read_parquet(nodes_path)
                
                # 时间过滤
                if t0 is not None and t_hi is not None:
                    nodes_df = nodes_df[(nodes_df["t"] >= t0) & (nodes_df["t"] <= t_hi)]
                
                # 重命名列以匹配期望格式
                if 'zone' in nodes_df.columns and 'supply' in nodes_df.columns:
                    return nodes_df[['zone', 't', 'soc', 'supply']].copy()
            
            # 如果solver_graph数据不存在，尝试从intermediate数据加载
            intermediate_path = Path(f"{self.inter_dir}/initial_inventory.parquet")
            if intermediate_path.exists():
                inventory_df = pd.read_parquet(intermediate_path)
                
                # 时间过滤
                if t0 is not None and t_hi is not None:
                    inventory_df = inventory_df[(inventory_df["t"] >= t0) & (inventory_df["t"] <= t_hi)]
                
                # 重命名列：count -> supply
                if 'count' in inventory_df.columns:
                    inventory_df = inventory_df.rename(columns={'count': 'supply'})
                
                return inventory_df
            
            # 如果都没有，返回空DataFrame
            print("[reposition_arc] Warning: No current inventory data found")
            return pd.DataFrame(columns=['zone', 't', 'soc', 'supply'])
            
        except Exception as e:
            print(f"[reposition_arc] Error loading current inventory: {e}")
            return pd.DataFrame(columns=['zone', 't', 'soc', 'supply'])
    
    def generate_arcs(self, 
                     reachable_set: Set[Tuple[int, int, int]],
                     t0: Optional[int] = None,
                     t_hi: Optional[int] = None,
                     B: Optional[int] = None) -> List[ArcMetadata]:
        """
        生成reposition弧：(i,t,l) -> (j,t+tau,l-de)
        """
        # 计算重定位需求
        reposition_demand = self._compute_reposition_demand(t0, t_hi)
        
        if reposition_demand.empty:
            return []
        
        # 加载距离和时间信息
        bij = load_base_ij(self.cfg)
        
        # 合并重定位需求与距离信息
        demand_with_dist = reposition_demand.merge(bij, on=['i', 'j'], how='inner')
        
        if demand_with_dist.empty:
            return []
        
        # 生成弧的逻辑
        print(f"[reposition_arc] Generating arcs from {len(demand_with_dist)} demand records")
        arcs = self._generate_reposition_arcs_from_pairs(reachable_set, demand_with_dist, t0, t_hi, B)
        print(f"[reposition_arc] Generated {len(arcs)} reposition arcs")
        return arcs
    
    def _generate_reposition_arcs_from_pairs(self, 
                                           reachable_set: Set[Tuple[int, int, int]],
                                           od_pairs_df: pd.DataFrame, 
                                           t0: Optional[int] = None, 
                                           t_hi: Optional[int] = None, 
                                           B: Optional[int] = None) -> List[ArcMetadata]:
        """
        从OD对生成重定位弧的通用函数
        """
        arrival_end = min(t_hi + int(B) if (t_hi is not None and B is not None) else self.gi.times[-1], self.gi.times[-1])
        soc_levels = np.array(self.gi.socs, dtype=int)
        
        max_rep_min = float(self.cfg.pruning.max_reposition_tt)
        max_rep_steps = int(math.ceil(max_rep_min / self.dt_minutes)) if max_rep_min > 0 else None
        
        arcs = []
        
        for _, r in od_pairs_df.iterrows():
            i, j, t = int(r["i"]), int(r["j"]), int(r["t"])
            
            # 重定位要求 i != j（不能重定位到同一区域）
            if i == j:
                continue
                
            dist_km = float(r.get("dist_km", 0.0))
            
            # 基于距离和速度计算耗时
            tau = self._compute_travel_time(dist_km)
            
            if tau <= 0:
                continue
            if max_rep_steps is not None and tau > max_rep_steps:
                continue
                
            t2 = t + tau
            if t2 > arrival_end:
                continue
                
            de = self._compute_energy_consumption(dist_km, t, tau, "de_per_km_rep")
            lmin = max(int(self.cfg.pruning.min_soc_for_reposition), de)
            feas_soc = soc_levels[soc_levels >= lmin]
            
            for l in feas_soc:
                if not self._check_reachability(i, l, t, reachable_set):
                    continue
                if not self._check_reachability(j, l - de, t2, reachable_set):
                    continue
                    
                from_id = self.gi.id_of(i, t, l)
                to_id = self.gi.id_of(j, t2, l - de)
                
                arc = ArcMetadata(
                    arc_type="reposition",
                    from_node_id=from_id,
                    to_node_id=to_id,
                    i=i,
                    j=j,
                    t=t,
                    l=l,
                    tau=tau,
                    de=de,
                    dist_km=dist_km
                )
                arcs.append(arc)
        
        return arcs
    
    # 注意：compute_costs方法已移除，成本计算统一由ArcAssembly.compute_costs_batch处理


# 注册到工厂
ArcFactory.register_arc_type("reposition", RepositionArc)


def main():
    """测试reposition弧生成"""
    cfg = get_config()
    gi = load_indexer()
    
    # 加载可达性
    from arcs.arc_base import load_reachability_with_time
    reachable_set = load_reachability_with_time()
    
    # 创建reposition弧生成器
    reposition_generator = RepositionArc(cfg, gi)
    
    # 生成弧
    arcs = reposition_generator.generate_arcs(reachable_set)
    
    print(f"生成了 {len(arcs)} 条reposition弧")
    
    # 转换为DataFrame并保存
    df = reposition_generator.to_dataframe(arcs)
    
    output_dir = Path(self.inter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "reposition_arcs_new.parquet"
    df = reposition_generator.generate_and_save(reachable_set, output_path)
    
    print(f"保存到: {output_path}")
    print(f"弧数量: {len(df)}")
    
    if not df.empty:
        print("\n弧类型分布:")
        print(df['arc_type'].value_counts())
        
        print("\n前5条弧:")
        print(df.head())


if __name__ == "__main__":
    main()
