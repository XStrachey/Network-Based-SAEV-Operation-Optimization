# charging_arc.py
# Charging弧的具体实现
from __future__ import annotations

import json
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
from utils.data_loader import (
    load_stations_mapping,
    load_or_build_charging_profile,
    load_station_capacity_map,
)


class ChargingArc(ArcBase):
    """Charging弧实现 - 纯网络化的充电弧（占用链）"""
    
    @property
    def arc_type_name(self) -> str:
        return "charging"
    
    def _load_zone_station_data(self) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """
        读取充电站数据：
        - zone_station_best.parquet: 列含 i,k,dist_km,tau_steps
        - nearest_stations.json: {i: [k1,k2,...]}
        """
        out_dir = Path("data/intermediate")
        best_path = out_dir / "zone_station_best.parquet"
        nearest_path = out_dir / "nearest_stations.json"
        
        if not best_path.exists():
            raise FileNotFoundError(f"zone_station_best.parquet not found at {best_path}. 请先运行 reachability.py")
        if not nearest_path.exists():
            raise FileNotFoundError(f"nearest_stations.json not found at {nearest_path}. 请先运行 reachability.py")
        
        best_df = pd.read_parquet(best_path)
        with open(nearest_path, 'r') as f:
            nearest_data = json.load(f)
        nearest_map = {int(k): v for k, v in nearest_data.items()}
        
        return best_df, nearest_map
    
    def generate_arcs(self, 
                     reachable_set: Set[Tuple[int, int, int]],
                     t0: Optional[int] = None,
                     t_hi: Optional[int] = None,
                     B: Optional[int] = None) -> List[ArcMetadata]:
        """
        生成charging弧：
        - tochg:     (i,t,l) -> (zone_k, t+tau_to, l - de_to)
        - chg_enter: (zone_k, p, lq) -> (q_in[k,p])，tau=0
        - chg_occ:   (q_in[k,p]) -> (q_out[k,p])，唯一容量弧（cap_hint = plugs_kp；成本=充电成本-奖励 由 costs 附加）
        - chg_step:  (q_out[k,p]) -> (zone_k, p+1, lq')，若为最后一步则 lq' = target_soc，否则 lq'=lq（保持 SOC）
        """
        best_df, nearest_map = self._load_zone_station_data()
        k2zone, k2level = load_stations_mapping(self.cfg)
        prof = load_or_build_charging_profile(self.cfg)
        
        # 最近站过滤
        pairs = [(int(i), int(k)) for i, ks in nearest_map.items() for k in ks]
        if pairs:
            near_df = pd.DataFrame(pairs, columns=["i", "k"])
            i2k = best_df.merge(near_df, on=["i", "k"], how="inner")
        else:
            i2k = best_df.iloc[0:0][["i", "k", "dist_km", "tau_steps"]].copy()
        i2k = i2k[i2k["k"].isin(k2zone.keys())].copy()
        
        arrival_end = min(t_hi + int(B) if (t_hi is not None and B is not None) else self.gi.times[-1], self.gi.times[-1])
        soc_levels = np.array(self.gi.socs, dtype=int)
        soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100
        
        # profile: (level) -> {(from_soc, to_soc): tau_minutes}
        prof_map: Dict[int, Dict[Tuple[int, int], float]] = {}
        for level, sub in prof.groupby("level"):
            d = {(int(r["from_soc"]), int(r["to_soc"])): float(r["tau_chg_minutes"]) for _, r in sub.iterrows()}
            prof_map[int(level)] = d
        
        min_step = int(self.cfg.charge_queue.min_charge_step)
        default_plugs = int(self.cfg.charge_queue.default_plugs_per_station)
        
        cap_map = load_station_capacity_map(self.cfg)  # k -> 有效并发
        
        arcs = []
        # 时间范围
        if t0 is not None:
            time_range = range(t0, t_hi)
        else:
            time_range = self.gi.times
        
        for _, r in i2k.iterrows():
            i, k = int(r["i"]), int(r["k"])
            dist_km = float(r.get("dist_km", 0.0))
            tau_to = int(r["tau_steps"])
            level_k = int(k2level[k])
            zone_k = int(k2zone[k])
            
            for t in time_range:
                t_arr = t + tau_to
                if t_arr > arrival_end:
                    continue
                de_to = self._compute_energy_consumption(dist_km, t, tau_to, "de_per_km_tochg")
                
                for l in soc_levels:
                    if l < de_to:
                        continue
                    if not self._check_reachability(i, int(l), t, reachable_set):
                        continue
                    
                    # 添加SOC限制：电量高于60的节点不允许生成充电弧
                    if l > 60:
                        continue
                    
                    # 简化模型：充电目标电量统一设置为100%
                    soc_after_travel = int(l) - int(de_to)  # 到站后的SOC
                    
                    # 目标SOC统一设置为100%
                    target_soc = 100
                    
                    # 查询充电时间（分钟 -> 步）——从到站 SOC 充到 target_soc
                    tau_chg_min = prof_map.get(level_k, {}).get((int(soc_after_travel), int(target_soc)), None)
                    if tau_chg_min is None:
                        # profile 未覆盖时可退化为线性或跳过
                        continue
                    tau_chg = int(math.ceil(tau_chg_min / self.dt_minutes))
                    if tau_chg <= 0:
                        print(f"tau_chg <= 0: {tau_chg}")
                        continue
                    
                    t_end = t_arr + tau_chg
                    if t_end > arrival_end:
                        continue
                    
                    # 1) 到站弧 tochg - 使用专门的充电站到达节点
                    if not self._check_reachability(zone_k, int(l - de_to), t_arr, reachable_set):
                        continue
                    from_id = self.gi.id_of(i, t, int(l))
                    # 创建专门的充电站到达节点，避免与网格节点冲突
                    chg_arrival_node = self._pseudo_node_id("chg_arrival", k, t_arr, int(l - de_to))
                    
                    tochg_arc = ArcMetadata(
                        arc_type="tochg",
                        from_node_id=from_id,
                        to_node_id=chg_arrival_node,
                        i=i,
                        k=k,
                        t=t,
                        l=int(l),
                        l_to=int(l - de_to),
                        tau=tau_to,
                        de=int(de_to),
                        level=level_k,
                        dist_km=dist_km
                    )
                    arcs.append(tochg_arc)
                    
                    # 2) 占用链：p = t_arr ... t_end-1
                    soc_now = soc_after_travel  # 使用修正后的到站SOC
                    p = t_arr
                    for step in range(tau_chg):
                        q_in = self._pseudo_node_id("q_in", k, p)
                        q_out = self._pseudo_node_id("q_out", k, p)
                        
                        # 检查当前时间片的可达性
                        if not self._check_reachability(zone_k, soc_now, p, reachable_set):
                            break
                        
                        # 2.1 chg_enter： 从专门的充电站到达节点进入充电站队列
                        # 这确保车辆必须通过tochg弧才能进入充电站
                        # 只有在第一步(p = t_arr)时从充电站到达节点进入
                        if step == 0:  # 只在第一步生成进入弧
                            enter_arc = ArcMetadata(
                                arc_type="chg_enter",
                                from_node_id=chg_arrival_node,
                                to_node_id=q_in,
                                i=i,
                                k=k,
                                t=p,
                                l=soc_now,
                                l_to=soc_now,
                                tau=0,
                                de=0,
                                level=level_k
                            )
                            arcs.append(enter_arc)
                        
                        # 2.2 chg_occ： q_in[k,p] -> q_out[k,p]  （唯一容量弧；cap_hint=plugs_kp）
                        plugs_kp = int(cap_map.get(k, default_plugs))  # ← 用站点并发替换默认值
                        occ_arc = ArcMetadata(
                            arc_type="chg_occ",
                            from_node_id=q_in,
                            to_node_id=q_out,
                            i=i,
                            k=k,
                            t=p,
                            l=soc_now,
                            l_to=soc_now,
                            tau=0,
                            de=0,
                            level=level_k,
                            cap_hint=plugs_kp,
                            tau_tochg=tau_to,  # 添加去充电时间
                            tau_chg=tau_chg    # 添加充电时间
                        )
                        arcs.append(occ_arc)
                        
                        # 2.3 chg_step： q_out[k,p] -> q_in[k,p+1] 或 -> (zone_k, p+1, soc_next)
                        is_last = (step == tau_chg - 1)
                        soc_next = int(target_soc) if is_last else int(soc_now)
                        
                        if is_last:
                            # 最后一步：从充电站队列回到网格节点
                            if not self._check_reachability(zone_k, soc_next, p+1, reachable_set):
                                break
                            to_id = self.gi.id_of(zone_k, p+1, soc_next)
                        else:
                            # 中间步骤：连接到下一个时间片的充电站队列
                            to_id = self._pseudo_node_id("q_in", k, p+1)
                        
                        step_arc = ArcMetadata(
                            arc_type="chg_step",
                            from_node_id=q_out,
                            to_node_id=to_id,
                            i=i,
                            k=k,
                            t=p,
                            l=soc_now,
                            l_to=soc_next,
                            tau=1,
                            de=0,
                            level=level_k,
                            is_last_step=is_last
                        )
                        arcs.append(step_arc)
                        
                        # 更新到下一个时间片
                        soc_now = soc_next
                        p += 1
                    
                    # 注意：不需要额外的出站弧！
                    # chg_step弧已经将车辆从充电站队列带回到网格节点(zone_k, p+1, soc_next)
                    # 车辆可以从网格节点继续通过idle、reposition、service等弧流动
        
        return arcs
    
    # 注意：compute_costs方法已移除，成本计算统一由ArcAssembly.compute_costs_batch处理


# 注册到工厂
ArcFactory.register_arc_type("charging", ChargingArc)


def main():
    """测试charging弧生成"""
    cfg = get_config()
    gi = load_indexer()
    
    # 加载可达性
    from arcs.arc_base import load_reachability_with_time
    reachable_set = load_reachability_with_time()
    
    # 创建charging弧生成器
    charging_generator = ChargingArc(cfg, gi)
    
    # 生成弧
    arcs = charging_generator.generate_arcs(reachable_set)
    
    print(f"生成了 {len(arcs)} 条charging弧")
    
    # 转换为DataFrame并保存
    df = charging_generator.to_dataframe(arcs)
    
    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "charging_arcs_new.parquet"
    df = charging_generator.generate_and_save(reachable_set, output_path)
    
    print(f"保存到: {output_path}")
    print(f"弧数量: {len(df)}")
    
    if not df.empty:
        print("\n弧类型分布:")
        print(df['arc_type'].value_counts())
        
        print("\n前5条弧:")
        print(df.head())


if __name__ == "__main__":
    main()
