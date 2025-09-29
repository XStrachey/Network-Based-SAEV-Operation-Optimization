# grid_utils.py
# 网格索引器和相关工具函数的共用底层逻辑
# 为 reachability 和 arc_generators 提供统一的网格操作接口
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GridIndexers:
    """
    网格索引器类，用于管理时空SOC网格的索引映射
    
    Attributes:
        zones: 区域列表
        times: 时间步列表  
        socs: SOC级别列表
        zone_to_idx: 区域到索引的映射
        time_to_idx: 时间到索引的映射
        soc_to_idx: SOC到索引的映射
        n_z: 区域数量
        n_t: 时间步数量
        n_l: SOC级别数量
    """
    zones: List[int]
    times: List[int]
    socs: List[int]
    zone_to_idx: Dict[int, int]
    time_to_idx: Dict[int, int]
    soc_to_idx: Dict[int, int]
    n_z: int
    n_t: int
    n_l: int

    def id_of(self, zone: int, t: int, soc: int) -> int:
        """
        根据区域、时间、SOC计算节点ID
        
        Args:
            zone: 区域ID
            t: 时间步
            soc: SOC级别
            
        Returns:
            节点ID
        """
        zi = self.zone_to_idx[zone]
        ti = self.time_to_idx[t]
        li = self.soc_to_idx[soc]
        return (zi * self.n_t + ti) * self.n_l + li

    def tuple_of(self, node_id: int) -> Tuple[int, int, int]:
        """
        根据节点ID计算区域、时间、SOC
        
        Args:
            node_id: 节点ID
            
        Returns:
            (区域ID, 时间步, SOC级别)
        """
        zi = node_id // (self.n_t * self.n_l)
        rem = node_id % (self.n_t * self.n_l)
        ti = rem // self.n_l
        li = rem % self.n_l
        return (self.zones[zi], self.times[ti], self.socs[li])

    @property
    def num_nodes(self) -> int:
        """返回总节点数量"""
        return self.n_z * self.n_t * self.n_l

def load_indexer(inter_dir: str = "data/intermediate") -> GridIndexers:
    """
    从pickle文件加载网格索引器
    
    Args:
        inter_dir: 中间文件目录路径
        
    Returns:
        GridIndexers对象
    """
    with open(Path(f"{inter_dir}/node_indexer.pkl"), "rb") as f:
        return pickle.load(f)

def ensure_dir(p: Path) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        p: 目录路径
    """
    p.mkdir(parents=True, exist_ok=True)

def first_present(columns, candidates: List[str]) -> Optional[str]:
    """
    从候选列名中找到第一个在列名列表中存在的列（不区分大小写）
    
    Args:
        columns: 列名列表
        candidates: 候选列名列表
        
    Returns:
        找到的列名，如果没找到则返回None
    """
    lowers = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowers:
            return lowers[cand.lower()]
    return None

def validate_grid_indexer(gi: GridIndexers) -> Dict[str, bool]:
    """
    验证网格索引器的完整性
    
    Args:
        gi: 网格索引器对象
        
    Returns:
        验证结果字典
    """
    results = {}
    
    # 检查基本属性
    results["has_zones"] = len(gi.zones) > 0
    results["has_times"] = len(gi.times) > 0
    results["has_socs"] = len(gi.socs) > 0
    
    # 检查索引映射
    results["zone_mapping_complete"] = len(gi.zone_to_idx) == len(gi.zones)
    results["time_mapping_complete"] = len(gi.time_to_idx) == len(gi.times)
    results["soc_mapping_complete"] = len(gi.soc_to_idx) == len(gi.socs)
    
    # 检查维度一致性
    results["dimensions_consistent"] = (
        gi.n_z == len(gi.zones) and
        gi.n_t == len(gi.times) and
        gi.n_l == len(gi.socs)
    )
    
    # 检查节点ID计算的一致性
    results["node_id_consistent"] = True
    try:
        for zone in gi.zones[:3]:  # 只检查前3个区域以节省时间
            for t in gi.times[:3]:  # 只检查前3个时间步
                for soc in gi.socs[:3]:  # 只检查前3个SOC级别
                    node_id = gi.id_of(zone, t, soc)
                    zone_back, t_back, soc_back = gi.tuple_of(node_id)
                    if zone_back != zone or t_back != t or soc_back != soc:
                        results["node_id_consistent"] = False
                        break
                if not results["node_id_consistent"]:
                    break
            if not results["node_id_consistent"]:
                break
    except Exception:
        results["node_id_consistent"] = False
    
    return results

def generate_grid_summary(gi: GridIndexers) -> Dict[str, any]:
    """
    生成网格摘要信息
    
    Args:
        gi: 网格索引器对象
        
    Returns:
        网格摘要字典
    """
    return {
        "total_zones": gi.n_z,
        "total_times": gi.n_t,
        "total_soc_levels": gi.n_l,
        "total_nodes": gi.num_nodes,
        "zone_range": (min(gi.zones), max(gi.zones)) if gi.zones else None,
        "time_range": (min(gi.times), max(gi.times)) if gi.times else None,
        "soc_range": (min(gi.socs), max(gi.socs)) if gi.socs else None,
        "validation": validate_grid_indexer(gi)
    }

# 公共包装函数，遵循用户偏好
def generate_grid_indexer() -> GridIndexers:
    """
    公共包装函数：加载网格索引器
    
    Returns:
        GridIndexers对象
    """
    return load_indexer()

def generate_grid_summary_report(gi: GridIndexers = None) -> Dict[str, any]:
    """
    公共包装函数：生成网格摘要报告
    
    Args:
        gi: 网格索引器对象，默认为None时自动加载
        
    Returns:
        网格摘要报告
    """
    if gi is None:
        gi = load_indexer()
    
    return generate_grid_summary(gi)

def generate_node_id(zone: int, t: int, soc: int, gi: GridIndexers = None) -> int:
    """
    公共包装函数：计算节点ID
    
    Args:
        zone: 区域ID
        t: 时间步
        soc: SOC级别
        gi: 网格索引器对象，默认为None时自动加载
        
    Returns:
        节点ID
    """
    if gi is None:
        gi = load_indexer()
    
    return gi.id_of(zone, t, soc)

def generate_node_tuple(node_id: int, gi: GridIndexers = None) -> Tuple[int, int, int]:
    """
    公共包装函数：根据节点ID计算区域、时间、SOC
    
    Args:
        node_id: 节点ID
        gi: 网格索引器对象，默认为None时自动加载
        
    Returns:
        (区域ID, 时间步, SOC级别)
    """
    if gi is None:
        gi = load_indexer()
    
    return gi.tuple_of(node_id)
