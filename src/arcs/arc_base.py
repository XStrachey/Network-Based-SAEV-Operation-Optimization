# arc_base.py
# 网络弧基类定义 - 抽象底层和具体层
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from config.network_config import get_network_config as get_config
from utils.grid_utils import GridIndexers, load_indexer


@dataclass
class ArcCost:
    """弧成本信息"""
    coef_rep: float = 0.0          # 重定位成本
    coef_chg_travel: float = 0.0   # 充电行驶成本
    coef_chg_occ: float = 0.0      # 充电占用成本
    coef_svc_gate: float = 0.0     # 服务闸门成本（负值）
    coef_rep_reward: float = 0.0   # 重定位奖励（负值）
    coef_chg_reward: float = 0.0   # 充电奖励（负值）
    coef_idle: float = 0.0         # idle机会成本
    coef_total: float = 0.0        # 总成本
    
    def __post_init__(self):
        """计算总成本"""
        self.coef_total = (
            self.coef_rep + self.coef_chg_travel + self.coef_chg_occ +
            self.coef_svc_gate + self.coef_rep_reward + self.coef_chg_reward + self.coef_idle
        )


@dataclass
class ArcMetadata:
    """弧元数据信息"""
    arc_type: str
    from_node_id: int
    to_node_id: int
    i: Optional[int] = None      # 起始区域
    j: Optional[int] = None      # 目标区域
    k: Optional[int] = None      # 充电站ID
    t: Optional[int] = None      # 起始时间
    l: Optional[int] = None      # 起始SOC
    l_to: Optional[int] = None   # 目标SOC
    tau: Optional[int] = None    # 时间步长
    de: Optional[int] = None     # 能耗
    dist_km: Optional[float] = None  # 距离
    level: Optional[int] = None  # 充电站等级
    cap_hint: Optional[int] = None   # 容量提示
    req_key: Optional[str] = None    # 需求键
    is_last_step: Optional[bool] = None  # 是否最后一步
    tau_tochg: Optional[int] = None  # 去充电时间（用于充电占用成本计算）
    tau_chg: Optional[int] = None    # 充电时间（用于充电占用成本计算）
    cost: Optional[ArcCost] = None   # 成本信息
    
    def __post_init__(self):
        """生成稳定的弧ID"""
        if hasattr(self, 'arc_id') and self.arc_id is not None:
            return
        
        # 构建唯一键
        key_parts = [self.arc_type, str(self.from_node_id), str(self.to_node_id)]
        if self.i is not None:
            key_parts.append(str(self.i))
        if self.j is not None:
            key_parts.append(str(self.j))
        if self.k is not None:
            key_parts.append(str(self.k))
        if self.t is not None:
            key_parts.append(str(self.t))
        if self.l is not None:
            key_parts.append(str(self.l))
        
        key = "|".join(key_parts)
        # 使用blake2b生成稳定的ID
        digest = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
        self.arc_id = int.from_bytes(digest, byteorder='big', signed=False) & 0x7FFFFFFFFFFFFFFF


class ArcBase(ABC):
    """弧基类 - 定义所有弧类型的通用接口和行为"""
    
    def __init__(self, cfg=None, gi: GridIndexers = None):
        self.cfg = cfg or get_config()
        self.gi = gi or load_indexer()
        self.dt_minutes = int(self.cfg.time_soc.dt_minutes)
        
    @property
    @abstractmethod
    def arc_type_name(self) -> str:
        """弧类型名称"""
        pass
    
    @abstractmethod
    def generate_arcs(self, 
                     reachable_set: Set[Tuple[int, int, int]],
                     t0: Optional[int] = None,
                     t_hi: Optional[int] = None,
                     B: Optional[int] = None) -> List[ArcMetadata]:
        """生成弧列表
        
        Args:
            reachable_set: 可达的(zone, soc, time)集合
            t0: 窗口起始时间
            t_hi: 窗口结束时间
            B: Halo步数
            
        Returns:
            弧元数据列表
        """
        pass
    
    def _pseudo_node_id(self, kind: str, *keys) -> int:
        """生成稳定的伪节点ID（负数），避免与网格节点冲突"""
        s = f"{kind}|" + "|".join(map(str, keys))
        digest = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
        val = int.from_bytes(digest, byteorder='big', signed=False) & 0x7FFFFFFFFFFFFFFF
        return -int(val)
    
    def _is_self_loop(self, from_node_id: int, to_node_id: int) -> bool:
        """检查是否为自环"""
        return from_node_id == to_node_id
    
    def _filter_self_loops(self, arcs: List[ArcMetadata]) -> List[ArcMetadata]:
        """过滤掉自环"""
        filtered = []
        self_loop_count = 0
        
        for arc in arcs:
            if self._is_self_loop(arc.from_node_id, arc.to_node_id):
                self_loop_count += 1
                if self.cfg.solver.verbose:
                    print(f"[{self.arc_type_name}] 跳过自环: {arc.from_node_id} -> {arc.to_node_id}")
            else:
                filtered.append(arc)
        
        if self_loop_count > 0 and self.cfg.solver.verbose:
            print(f"[{self.arc_type_name}] 过滤了 {self_loop_count} 个自环")
            
        return filtered
    
    def _check_reachability(self, zone: int, soc: int, time: int, 
                          reachable_set: Set[Tuple[int, int, int]]) -> bool:
        """检查节点是否可达"""
        return (zone, soc, time) in reachable_set
    
    def _compute_travel_time(self, dist_km: float) -> int:
        """基于距离计算旅行时间（步数）"""
        avg_speed_kmh = float(self.cfg.basic.avg_speed_kmh)
        travel_time_minutes = (dist_km / avg_speed_kmh) * 60.0 if avg_speed_kmh > 0 else 0.0
        return int(math.ceil(travel_time_minutes / self.dt_minutes))
    
    def _compute_energy_consumption(self, dist_km: float, t: int, tau: int, 
                                  energy_type: str) -> int:
        """计算能耗"""
        from utils.energy_utils import compute_multi_timestep_energy_consumption
        soc_levels = np.array(self.gi.socs, dtype=int)
        soc_step = int(np.diff(soc_levels).min()) if len(soc_levels) > 1 else 100
        return compute_multi_timestep_energy_consumption(
            self.cfg, dist_km, t, tau, soc_step, energy_type
        )
    
    def to_dataframe(self, arcs: List[ArcMetadata]) -> pd.DataFrame:
        """将弧列表转换为DataFrame"""
        if not arcs:
            return pd.DataFrame()
        
        rows = []
        for arc in arcs:
            row = {
                'arc_id': arc.arc_id,
                'arc_type': arc.arc_type,
                'from_node_id': arc.from_node_id,
                'to_node_id': arc.to_node_id,
            }
            
            # 添加可选字段
            optional_fields = ['i', 'j', 'k', 't', 'l', 'l_to', 'tau', 'de', 
                             'dist_km', 'level', 'cap_hint', 'req_key', 'is_last_step',
                             'tau_tochg', 'tau_chg']
            for field in optional_fields:
                value = getattr(arc, field, None)
                if value is not None:
                    row[field] = value
            
            # 添加成本信息
            if arc.cost is not None:
                row.update({
                    'coef_rep': arc.cost.coef_rep,
                    'coef_chg_travel': arc.cost.coef_chg_travel,
                    'coef_chg_occ': arc.cost.coef_chg_occ,
                    'coef_svc_gate': arc.cost.coef_svc_gate,
                    'coef_rep_reward': arc.cost.coef_rep_reward,
                    'coef_chg_reward': arc.cost.coef_chg_reward,
                    'coef_idle': arc.cost.coef_idle,
                    'coef_total': arc.cost.coef_total,
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    # 注意：compute_costs方法已移除，成本计算统一由ArcAssembly.compute_costs_batch处理
    
    def generate_and_save(self, 
                         reachable_set: Set[Tuple[int, int, int]],
                         output_path: Optional[Path] = None,
                         t0: Optional[int] = None,
                         t_hi: Optional[int] = None,
                         B: Optional[int] = None) -> pd.DataFrame:
        """生成弧并保存到文件（不计算成本，成本由ArcAssembly统一计算）"""
        arcs = self.generate_arcs(reachable_set, t0, t_hi, B)
        
        # 过滤自环
        arcs = self._filter_self_loops(arcs)
        
        # 注意：不在这里计算成本，成本由ArcAssembly.compute_costs_batch统一计算
        
        # 转换为DataFrame
        df = self.to_dataframe(arcs)
        
        # 保存到文件
        if output_path is not None:
            df.to_parquet(output_path, index=False)
            if self.cfg.solver.verbose:
                print(f"[{self.arc_type_name}] 保存了 {len(df)} 条弧到 {output_path}")
        
        return df


class ArcFactory:
    """弧工厂类 - 管理所有弧类型的创建"""
    
    _arc_classes = {}
    
    @classmethod
    def register_arc_type(cls, arc_type: str, arc_class):
        """注册弧类型"""
        cls._arc_classes[arc_type] = arc_class
    
    @classmethod
    def create_arc(cls, arc_type: str, cfg=None, gi: GridIndexers = None, inter_dir: str = "data/intermediate") -> ArcBase:
        """创建指定类型的弧"""
        if arc_type not in cls._arc_classes:
            raise ValueError(f"未知的弧类型: {arc_type}")
        
        return cls._arc_classes[arc_type](cfg, gi, inter_dir)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取所有可用的弧类型"""
        return list(cls._arc_classes.keys())


def load_reachability_with_time(path="data/intermediate/reachability.parquet") -> Set[Tuple[int, int, int]]:
    """加载可达性集合"""
    if not Path(path).exists():
        return set()
    
    df = pd.read_parquet(path)
    return set(
        (int(r["zone"]), int(r["soc"]), int(r["t"]))
        for _, r in df.iterrows()
        if r["reachable"] == 1
    )


class CoeffProvider:
    """
    提供时间可变成本系数 γ_rep_p(t), β_chg_p1(t), β_chg_p2(t), tou_price(t) 以及 VOT。
    优先从 cfg.paths.coeff_schedule 读取（列: t, gamma_rep_p, beta_chg_p1, beta_chg_p2, tou_price），
    若缺失则回退到 config.costs_equity 的常数。
    """
    def __init__(self, schedule_path: Optional[str] = None):
        cfg = get_config()
        self.dt_minutes = int(cfg.time_soc.dt_minutes)

        ce = cfg.costs_equity
        self.vot = float(ce.vot)
        self.gamma_rep_p_const = float(ce.gamma_rep)     # 重定位时间成本
        self.beta_chg_p1_const = float(ce.beta_toCHG)    # 去充电行驶成本
        self.beta_chg_p2_const = float(ce.beta_chg)      # 充电占位成本
        self.tou_price_const = 0.0                       # 默认TOU价格为0（FCFS）

        if schedule_path is None:
            schedule_path = cfg.paths.coeff_schedule

        self.has_schedule = Path(schedule_path).exists()
        self.map_gamma_rep_p: Dict[int, float] = {}
        self.map_beta_chg_p1: Dict[int, float] = {}
        self.map_beta_chg_p2: Dict[int, float] = {}
        self.map_tou_price: Dict[int, float] = {}
        if self.has_schedule:
            sch = pd.read_csv(schedule_path)
            if "t" not in sch.columns:
                raise ValueError("coeff_schedule.csv 需要列 't'")
            sch = sch.copy()
            if "gamma_rep_p" not in sch.columns: sch["gamma_rep_p"] = self.gamma_rep_p_const
            if "beta_chg_p1" not in sch.columns: sch["beta_chg_p1"] = self.beta_chg_p1_const
            if "beta_chg_p2" not in sch.columns: sch["beta_chg_p2"] = self.beta_chg_p2_const
            if "tou_price" not in sch.columns: sch["tou_price"] = self.tou_price_const
            sch["t"] = sch["t"].astype(int)
            self.map_gamma_rep_p = dict(zip(sch["t"], sch["gamma_rep_p"].astype(float)))
            self.map_beta_chg_p1 = dict(zip(sch["t"], sch["beta_chg_p1"].astype(float)))
            self.map_beta_chg_p2 = dict(zip(sch["t"], sch["beta_chg_p2"].astype(float)))
            self.map_tou_price = dict(zip(sch["t"], sch["tou_price"].astype(float)))

    # 点值
    def gamma_rep_p(self, t: int) -> float: return float(self.map_gamma_rep_p.get(int(t), self.gamma_rep_p_const))
    def beta_chg_p1(self, t: int) -> float: return float(self.map_beta_chg_p1.get(int(t), self.beta_chg_p1_const))
    def beta_chg_p2(self, t: int) -> float: return float(self.map_beta_chg_p2.get(int(t), self.beta_chg_p2_const))
    def tou_price(self, t: int) -> float: return float(self.map_tou_price.get(int(t), self.tou_price_const))

    # 区间累加
    def gamma_rep_p_sum_over_window(self, t_start: int, tau: int) -> float:
        tau = int(tau)
        if tau <= 0: return 0.0
        if not self.has_schedule: return tau * self.gamma_rep_p_const
        return float(sum(self.gamma_rep_p(tp) for tp in range(t_start, t_start + tau)))

    def beta_chg_p1_sum_over_window(self, t_start: int, tau: int) -> float:
        tau = int(tau)
        if tau <= 0: return 0.0
        if not self.has_schedule: return tau * self.beta_chg_p1_const
        return float(sum(self.beta_chg_p1(tp) for tp in range(t_start, t_start + tau)))

    def beta_chg_p2_sum_over_window(self, t_start: int, tau_tochg: int, tau_chg: int) -> float:
        tau_chg = int(tau_chg)
        if tau_chg <= 0: return 0.0
        if not self.has_schedule: return tau_chg * self.beta_chg_p2_const
        s_begin = int(t_start + tau_tochg)
        return float(sum(self.beta_chg_p2(tp) for tp in range(s_begin, s_begin + tau_chg)))
