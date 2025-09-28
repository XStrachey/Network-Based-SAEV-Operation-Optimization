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
    
    @property
    def arc_type_name(self) -> str:
        return "reposition"
    
    def _compute_reposition_demand(self, t0: Optional[int] = None, t_hi: Optional[int] = None) -> pd.DataFrame:
        """
        计算重定位需求矩阵
        
        基于当期目标区域车辆库存与未来4期平均需求的差额来计算重定位需求
        """
        # 加载服务需求数据
        from utils.data_loader import load_od_matrix
        od = load_od_matrix(self.cfg)
        
        # 时间过滤
        if t0 is not None and t_hi is not None:
            od = od[(od["t"] >= t0) & (od["t"] <= t_hi - 1)]
        
        if od.empty:
            return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
        
        # 加载当期车辆库存数据
        current_inventory = self._load_current_inventory(t0, t_hi)
        if current_inventory.empty:
            print("[reposition_arc] Warning: No current inventory data found")
            return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
        
        print(f"[reposition_arc] Loaded inventory data: {len(current_inventory)} records")
        print(f"[reposition_arc] Inventory time range: {current_inventory['t'].min()} - {current_inventory['t'].max()}")
        print(f"[reposition_arc] Total inventory: {current_inventory['supply'].sum()}")
        
        reposition_demand = []
        
        # 计算未来4期平均需求
        future_periods = 4
        all_times = sorted(od['t'].unique())
        
        # 为每个OD对计算未来4期平均需求
        od_future_avg = []
        for _, row in od.iterrows():
            i, j, t = int(row['i']), int(row['j']), int(row['t'])
            
            # 计算从 t+1 到 t+4 的未来需求
            future_demand_sum = 0.0
            valid_periods = 0
            
            for k in range(1, future_periods + 1):
                future_t = t + k
                if future_t in all_times:
                    future_dem = od[(od["t"] == future_t) & (od["i"] == i) & (od["j"] == j)]["demand"].sum()
                    future_demand_sum += future_dem
                    valid_periods += 1
            
            # 计算平均需求
            if valid_periods > 0:
                avg_future_demand = future_demand_sum / valid_periods
                od_future_avg.append({
                    'i': i,
                    'j': j,
                    't': t,
                    'demand': avg_future_demand
                })
        
        # 转换为DataFrame
        od_future = pd.DataFrame(od_future_avg)
        
        if od_future.empty:
            return pd.DataFrame(columns=['i', 'j', 't', 'reposition_demand'])
        
        # 计算每个区域在每个时刻的未来平均需求
        zone_demand = od_future.groupby(['i', 't'])['demand'].sum().reset_index()
        zone_demand.columns = ['zone', 't', 'future_avg_demand']
        
        # 计算每个区域在每个时刻的当期车辆库存
        zone_inventory = current_inventory.groupby(['zone', 't'])['supply'].sum().reset_index()
        zone_inventory.columns = ['zone', 't', 'current_inventory']
        
        # 合并需求和库存数据
        zone_balance = zone_demand.merge(zone_inventory, on=['zone', 't'], how='outer')
        zone_balance['future_avg_demand'] = zone_balance['future_avg_demand'].fillna(0.0)
        zone_balance['current_inventory'] = zone_balance['current_inventory'].fillna(0.0)
        
        # 计算供需差额：当期库存 - 未来4期平均需求
        zone_balance['demand_supply_gap'] = zone_balance['current_inventory'] - zone_balance['future_avg_demand']
        
        print(f"[reposition_arc] Zone balance analysis:")
        print(f"[reposition_arc] - Zones with supply surplus: {(zone_balance['demand_supply_gap'] > 0).sum()}")
        print(f"[reposition_arc] - Zones with demand surplus: {(zone_balance['demand_supply_gap'] < 0).sum()}")
        print(f"[reposition_arc] - Max supply surplus: {zone_balance['demand_supply_gap'].max():.2f}")
        print(f"[reposition_arc] - Max demand surplus: {abs(zone_balance['demand_supply_gap'].min()):.2f}")
        
        # 设置阈值 - 允许从有库存的区域重定位，即使供需平衡
        gap_threshold = getattr(self.cfg.pruning, 'reposition_gap_threshold', 0.0)  # 允许供需平衡的区域也重定位
        print(f"[reposition_arc] Using gap threshold: {gap_threshold}")
        
        # 生成重定位需求：从有库存的区域到需求过剩区域
        for _, row in zone_balance.iterrows():
            zone, t = int(row['zone']), int(row['t'])
            gap = row['demand_supply_gap']
            current_inventory = row['current_inventory']
            
            # 修改条件：有库存且供需差额大于等于阈值
            if current_inventory > 0 and gap >= gap_threshold:
                print(f"[reposition_arc] Found source zone {zone} at time {t} with inventory {current_inventory} and gap {gap}")
                
                # 找到需求过剩的区域作为目标（包括当前时间和未来时间）
                demand_surplus = zone_balance[
                    (zone_balance['t'] >= t) & 
                    (zone_balance['demand_supply_gap'] < -gap_threshold)
                ].copy()
                
                print(f"[reposition_arc] Found {len(demand_surplus)} demand surplus zones at time {t}")
                
                if not demand_surplus.empty:
                    # 选择需求过剩最严重的几个目标区域
                    top_targets = demand_surplus.nsmallest(3, 'demand_supply_gap')
                    
                    for _, target in top_targets.iterrows():
                        target_zone = int(target['zone'])
                        
                        # 重定位要求 i != j（不能重定位到同一区域）
                        if zone == target_zone:
                            print(f"[reposition_arc] Skipping same zone {zone}")
                            continue
                        
                        # 重定位需求 = 基于库存和需求过剩量的重定位
                        reposition_ratio = getattr(self.cfg.pruning, 'reposition_demand_ratio', 0.3)
                        
                        # 如果供需平衡，使用库存的一部分进行重定位
                        if gap >= 0:
                            # 供需平衡或供给过剩，使用库存的一部分
                            reposition_demand_val = current_inventory * reposition_ratio
                        else:
                            # 供给不足，使用供需差额和需求过剩量的较小值
                            reposition_demand_val = min(abs(gap), abs(target['demand_supply_gap'])) * reposition_ratio
                        
                        print(f"[reposition_arc] Reposition from {zone} to {target_zone}: demand_val={reposition_demand_val:.2f}")
                        
                        if reposition_demand_val > 0:
                            reposition_demand.append({
                                'i': zone,
                                'j': target_zone,
                                't': t,
                                'reposition_demand': reposition_demand_val
                            })
        
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
            print(demand_df)
        
        return demand_df
    
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
            intermediate_path = Path("data/intermediate/initial_inventory.parquet")
            if intermediate_path.exists():
                inventory_df = pd.read_parquet(intermediate_path)
                
                # 时间过滤
                if t0 is not None and t_hi is not None:
                    inventory_df = inventory_df[(inventory_df["t"] >= t0) & (inventory_df["t"] <= t_hi)]
                
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
    
    def compute_costs(self, arc: ArcMetadata) -> ArcCost:
        """计算reposition弧的成本"""
        # 重定位时间成本 - 使用正确的成本计算
        from arcs.arc_base import CoeffProvider
        cp = CoeffProvider(self.cfg.paths.coeff_schedule)
        
        vot = cp.vot
        gamma_rep = cp.gamma_rep_p_sum_over_window(arc.t, arc.tau)
        coef_rep = vot * gamma_rep
        
        # 重定位奖励（负成本）
        coef_rep_reward = 0.0
        if hasattr(self.cfg, 'flags') and getattr(self.cfg.flags, 'enable_reposition_reward', True):
            from config.costs import build_zone_value_table
            zone_val = build_zone_value_table()
            
            # 查找目标区域的zone_value
            target_value = zone_val[
                (zone_val['t'] == arc.t) & (zone_val['j'] == arc.j)
            ]['zone_value']
            
            if not target_value.empty:
                gamma_rep_a = float(self.cfg.costs_equity.gamma_reposition_reward)
                coef_rep_reward = -gamma_rep_a * float(target_value.iloc[0])
        
        return ArcCost(
            coef_rep=coef_rep,
            coef_rep_reward=coef_rep_reward,
            coef_chg_travel=0.0,
            coef_chg_occ=0.0,
            coef_svc_gate=0.0,
            coef_chg_reward=0.0,
            coef_idle=0.0
        )


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
    
    output_dir = Path("data/intermediate")
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
