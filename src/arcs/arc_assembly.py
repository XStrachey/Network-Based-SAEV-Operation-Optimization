# arc_assembly.py
# 弧组装脚本 - 整合所有弧类型，提供统一的网络构建接口
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# 确保src目录在Python路径中
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config.network_config import get_network_config as get_config
from utils.grid_utils import GridIndexers, load_indexer
from arcs.arc_base import ArcFactory, load_reachability_with_time

# 导入所有弧类型以注册到工厂
from arcs.idle_arc import IdleArc
from arcs.service_arc import ServiceArc
from arcs.reposition_arc import RepositionArc
from arcs.charging_arc import ChargingArc


class ArcAssembly:
    """弧组装器 - 管理所有弧类型的生成和整合"""
    
    def __init__(self, cfg=None, gi: GridIndexers = None, arc_types_override=None):
        self.cfg = cfg or get_config()
        self.gi = gi or load_indexer()
        self.reachable_set = load_reachability_with_time()
        
        # 弧类型配置
        self.arc_types = {
            "idle": {"generator": None, "enabled": True, "description": "车辆等待弧"},
            "service": {"generator": None, "enabled": True, "description": "服务需求弧"},
            "reposition": {"generator": None, "enabled": True, "description": "重定位弧"},
            "charging": {"generator": None, "enabled": True, "description": "充电弧"},
        }
        
        # 应用配置中的弧类型控制
        self._apply_arc_type_control()
        
        # 应用外部覆盖（如果提供）
        if arc_types_override is not None:
            self._apply_arc_types_override(arc_types_override)
        
        # 初始化弧生成器
        self._initialize_generators()
    
    def _apply_arc_type_control(self):
        """应用配置中的弧类型控制设置"""
        if hasattr(self.cfg, 'arc_control'):
            arc_control = self.cfg.arc_control
            self.arc_types["idle"]["enabled"] = arc_control.enable_idle
            self.arc_types["service"]["enabled"] = arc_control.enable_service
            self.arc_types["reposition"]["enabled"] = arc_control.enable_reposition
            self.arc_types["charging"]["enabled"] = arc_control.enable_charging
    
    def _apply_arc_types_override(self, arc_types_override):
        """应用外部提供的弧类型覆盖设置
        
        Args:
            arc_types_override: 字典，键为弧类型，值为是否启用
        """
        for arc_type, enabled in arc_types_override.items():
            if arc_type in self.arc_types:
                self.arc_types[arc_type]["enabled"] = enabled
    
    def _initialize_generators(self):
        """初始化所有弧生成器"""
        for arc_type in self.arc_types.keys():
            if self.arc_types[arc_type]["enabled"]:
                try:
                    generator = ArcFactory.create_arc(arc_type, self.cfg, self.gi)
                    self.arc_types[arc_type]["generator"] = generator
                    if self.cfg.solver.verbose:
                        print(f"[ArcAssembly] 初始化 {arc_type} 弧生成器")
                except Exception as e:
                    print(f"[ArcAssembly] 警告: 无法初始化 {arc_type} 弧生成器: {e}")
                    self.arc_types[arc_type]["enabled"] = False
    
    def set_arc_type_enabled(self, arc_type: str, enabled: bool):
        """启用或禁用特定弧类型"""
        if arc_type in self.arc_types:
            self.arc_types[arc_type]["enabled"] = enabled
            if enabled and self.arc_types[arc_type]["generator"] is None:
                try:
                    generator = ArcFactory.create_arc(arc_type, self.cfg, self.gi)
                    self.arc_types[arc_type]["generator"] = generator
                except Exception as e:
                    print(f"[ArcAssembly] 警告: 无法启用 {arc_type} 弧生成器: {e}")
                    self.arc_types[arc_type]["enabled"] = False
    
    def enable_only_arc_types(self, arc_types: List[str]):
        """只启用指定的弧类型，禁用其他所有类型
        
        Args:
            arc_types: 要启用的弧类型列表，如 ['idle', 'service']
        """
        # 首先禁用所有类型
        for arc_type in self.arc_types.keys():
            self.arc_types[arc_type]["enabled"] = False
        
        # 然后启用指定的类型
        for arc_type in arc_types:
            if arc_type in self.arc_types:
                self.set_arc_type_enabled(arc_type, True)
    
    def disable_arc_types(self, arc_types: List[str]):
        """禁用指定的弧类型
        
        Args:
            arc_types: 要禁用的弧类型列表
        """
        for arc_type in arc_types:
            if arc_type in self.arc_types:
                self.arc_types[arc_type]["enabled"] = False
    
    def get_enabled_arc_types(self) -> List[str]:
        """获取当前启用的弧类型列表"""
        return [arc_type for arc_type, config in self.arc_types.items() 
                if config["enabled"]]
    
    def get_arc_type_status(self) -> Dict[str, bool]:
        """获取所有弧类型的启用状态"""
        return {arc_type: config["enabled"] 
                for arc_type, config in self.arc_types.items()}
    
    def generate_all_arcs(self, 
                         t0: Optional[int] = None,
                         t_hi: Optional[int] = None,
                         B: Optional[int] = None,
                         save_individual: bool = True) -> Dict[str, pd.DataFrame]:
        """生成所有弧类型
        
        Args:
            t0: 窗口起始时间
            t_hi: 窗口结束时间
            B: Halo步数
            save_individual: 是否保存单独的弧文件
            
        Returns:
            字典，键为弧类型，值为对应的DataFrame
        """
        results = {}
        output_dir = Path("data/intermediate")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for arc_type, config in self.arc_types.items():
            if not config["enabled"] or config["generator"] is None:
                if self.cfg.solver.verbose:
                    print(f"[ArcAssembly] 跳过 {arc_type} 弧生成")
                results[arc_type] = pd.DataFrame()
                continue
            
            generator = config["generator"]
            
            if self.cfg.solver.verbose:
                print(f"[ArcAssembly] 生成 {arc_type} 弧...")
            
            try:
                # 生成弧
                if save_individual:
                    output_path = output_dir / f"{arc_type}_arcs_new.parquet"
                    df = generator.generate_and_save(
                        self.reachable_set, 
                        output_path, 
                        t0, t_hi, B
                    )
                else:
                    arcs_list = generator.generate_arcs(self.reachable_set, t0, t_hi, B)
                    df = generator.to_dataframe(arcs_list)
                
                results[arc_type] = df
                
                if self.cfg.solver.verbose:
                    print(f"[ArcAssembly] {arc_type} 弧: {len(df)} 条")
                    
            except Exception as e:
                print(f"[ArcAssembly] 错误: 生成 {arc_type} 弧失败: {e}")
                results[arc_type] = pd.DataFrame()
        
        return results
    
    def compute_costs_batch(self, arc_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """批量计算所有弧类型的成本"""
        for arc_type, df in arc_data.items():
            if df.empty or arc_type not in self.arc_types or not self.arc_types[arc_type]["enabled"]:
                continue
                
            generator = self.arc_types[arc_type]["generator"]
            if generator is None:
                continue
            
            # 初始化成本列
            df["coef_rep"] = 0.0
            df["coef_chg_travel"] = 0.0
            df["coef_chg_occ"] = 0.0
            df["coef_svc_gate"] = 0.0
            df["coef_rep_reward"] = 0.0
            df["coef_chg_reward"] = 0.0
            df["coef_idle"] = 0.0
                
            # 使用旧架构的批量成本计算逻辑
            if arc_type == "reposition":
                from arcs.arc_base import CoeffProvider
                cp = CoeffProvider(self.cfg.paths.coeff_schedule)
                
                # 重定位时间成本
                df["coef_rep"] = df.apply(
                    lambda r: cp.vot * cp.gamma_rep_p_sum_over_window(int(r["t"]), int(r["tau"])), axis=1
                )
                
                # 重定位奖励（如果有）
                if hasattr(self.cfg, 'flags') and getattr(self.cfg.flags, 'enable_reposition_reward', True):
                    from config.costs import build_zone_value_table
                    zone_val = build_zone_value_table()
                    
                    def compute_rep_reward(row):
                        target_value = zone_val[
                            (zone_val['t'] == row['t']) & (zone_val['j'] == row['j'])
                        ]['zone_value']
                        if not target_value.empty:
                            gamma_rep_a = float(self.cfg.costs_equity.gamma_reposition_reward)
                            return -gamma_rep_a * float(target_value.iloc[0])
                        return 0.0
                    
                    df["coef_rep_reward"] = df.apply(compute_rep_reward, axis=1)
                    
            elif arc_type == "charging":
                from arcs.arc_base import CoeffProvider
                cp = CoeffProvider(self.cfg.paths.coeff_schedule)
                
                # 去充电行驶成本 (tochg)
                tochg_mask = df["arc_type"] == "tochg"
                df.loc[tochg_mask, "coef_chg_travel"] = df.loc[tochg_mask].apply(
                    lambda r: cp.vot * cp.beta_chg_p1_sum_over_window(int(r["t"]), int(r["tau"])), axis=1
                )
                
                # 充电占用成本 (chg_occ)
                occ_mask = df["arc_type"] == "chg_occ"
                df.loc[occ_mask, "coef_chg_occ"] = df.loc[occ_mask].apply(
                    lambda r: cp.vot * cp.beta_chg_p2(int(r["t"])), axis=1
                )
                
                # 充电奖励（如果有）
                if hasattr(self.cfg, 'flags') and getattr(self.cfg.flags, 'enable_charging_reward', True):
                    beta_chg_a = float(self.cfg.costs_equity.beta_chg_reward)
                    
                    # chg_step奖励
                    step_mask = df["arc_type"] == "chg_step"
                    df.loc[step_mask, "coef_chg_reward"] = df.loc[step_mask].apply(
                        lambda r: -beta_chg_a * max(0, int(r.get("l_to", 0)) - int(r.get("l", 0))), axis=1
                    )
                    
            elif arc_type == "service":
                # 服务奖励（负成本）
                gate_mask = df["arc_type"] == "svc_gate"
                if gate_mask.any():
                    base_weight = float(self.cfg.costs_equity.unmet_weight_default)
                    vot = float(self.cfg.costs_equity.vot)
                    df.loc[gate_mask, "coef_svc_gate"] = -vot * base_weight
                    
            elif arc_type == "idle":
                # idle弧的成本是常数
                idle_cost = float(self.cfg.costs_equity.idle_opportunity_cost)
                df["coef_idle"] = idle_cost
            
            # 计算总成本
            df["coef_total"] = (
                df["coef_rep"] + df["coef_chg_travel"] + df["coef_chg_occ"] + 
                df["coef_svc_gate"] + df["coef_rep_reward"] + df["coef_chg_reward"] + 
                df["coef_idle"]
            )
            
            arc_data[arc_type] = df
            
        return arc_data
    
    def generate_for_window(self, t0: int, H: int, B: int) -> pd.DataFrame:
        """
        为给定时间窗口生成所有类型的弧（用于_08_build_solver_graph.py）
        """
        t_hi = min(int(self.cfg.time_soc.end_step), int(t0 + H))
        
        # 1. 生成所有弧类型（不计算成本）
        arc_data = self.generate_all_arcs(t0, t_hi, B, save_individual=False)
        
        # 2. 批量计算成本
        arc_data = self.compute_costs_batch(arc_data)
        
        # 3. 合并所有弧
        combined_df = self.combine_all_arcs(arc_data, save_combined=False)
        
        return combined_df
    
    def combine_all_arcs(self, 
                        arc_data: Dict[str, pd.DataFrame],
                        save_combined: bool = True) -> pd.DataFrame:
        """合并所有弧类型为统一的DataFrame
        
        Args:
            arc_data: 各弧类型的DataFrame字典
            save_combined: 是否保存合并后的文件
            
        Returns:
            合并后的弧DataFrame
        """
        all_arcs = []
        
        for arc_type, df in arc_data.items():
            if df is not None and not df.empty:
                all_arcs.append(df)
        
        if not all_arcs:
            combined_df = pd.DataFrame()
        else:
            combined_df = pd.concat(all_arcs, ignore_index=True)
        
        # 去重（基于arc_id）
        if not combined_df.empty and "arc_id" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["arc_id"]).sort_values("arc_id").reset_index(drop=True)
        
        if save_combined:
            output_dir = Path("data/intermediate")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "all_arcs_new.parquet"
            combined_df.to_parquet(output_path, index=False)
            
            if self.cfg.solver.verbose:
                print(f"[ArcAssembly] 合并弧保存到: {output_path}")
                print(f"[ArcAssembly] 总弧数量: {len(combined_df)}")
        
        return combined_df
    
    def get_statistics(self, arc_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """获取弧生成统计信息
        
        Args:
            arc_data: 各弧类型的DataFrame字典
            
        Returns:
            统计信息字典
        """
        stats = {
            "total_arcs": 0,
            "arc_type_counts": {},
            "cost_summary": {},
            "time_range": {
                "start": None,
                "end": None
            }
        }
        
        for arc_type, df in arc_data.items():
            if df is None or df.empty:
                stats["arc_type_counts"][arc_type] = 0
                continue
            
            count = len(df)
            stats["arc_type_counts"][arc_type] = count
            stats["total_arcs"] += count
            
            # 时间范围
            if "t" in df.columns:
                time_values = df["t"].dropna()
                if not time_values.empty:
                    if stats["time_range"]["start"] is None:
                        stats["time_range"]["start"] = int(time_values.min())
                        stats["time_range"]["end"] = int(time_values.max())
                    else:
                        stats["time_range"]["start"] = min(stats["time_range"]["start"], int(time_values.min()))
                        stats["time_range"]["end"] = max(stats["time_range"]["end"], int(time_values.max()))
            
            # 成本统计
            if "coef_total" in df.columns:
                cost_values = df["coef_total"].dropna()
                if not cost_values.empty:
                    stats["cost_summary"][arc_type] = {
                        "min": float(cost_values.min()),
                        "max": float(cost_values.max()),
                        "mean": float(cost_values.mean()),
                        "total": float(cost_values.sum())
                    }
        
        return stats
    
    def save_metadata(self, 
                     arc_data: Dict[str, pd.DataFrame],
                     output_dir: Optional[Path] = None):
        """保存弧生成元数据
        
        Args:
            arc_data: 各弧类型的DataFrame字典
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = Path("data/intermediate")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取统计信息
        stats = self.get_statistics(arc_data)
        
        # 添加配置信息
        metadata = {
            "generation_info": {
                "arc_types_enabled": [k for k, v in self.arc_types.items() if v["enabled"]],
                "total_arcs": stats["total_arcs"],
                "arc_type_counts": stats["arc_type_counts"],
                "time_range": stats["time_range"],
                "cost_summary": stats["cost_summary"],
            },
            "configuration": {
                "dt_minutes": self.cfg.time_soc.dt_minutes,
                "min_charge_step": self.cfg.charge_queue.min_charge_step,
                "default_plugs_per_station": self.cfg.charge_queue.default_plugs_per_station,
                "max_reposition_tt_minutes": self.cfg.pruning.max_reposition_tt,
                "min_soc_for_reposition": self.cfg.pruning.min_soc_for_reposition,
                "reposition_nearest_zone_n": self.cfg.pruning.reposition_nearest_zone_n,
            },
            "architecture": {
                "base_class": "ArcBase",
                "factory_pattern": True,
                "cost_integration": True,
                "self_loop_removal": True,
            }
        }
        
        metadata_path = output_dir / "arcs_metadata_new.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if self.cfg.solver.verbose:
            print(f"[ArcAssembly] 元数据保存到: {metadata_path}")


def main():
    """主函数 - 演示完整的弧生成流程"""
    cfg = get_config()
    gi = load_indexer()
    
    print("[ArcAssembly] 开始网络弧生成...")
    print(f"[ArcAssembly] 配置: {cfg.solver.verbose}")
    
    # 创建组装器
    assembly = ArcAssembly(cfg, gi)
    
    # 显示可用的弧类型
    available_types = ArcFactory.get_available_types()
    print(f"[ArcAssembly] 可用弧类型: {available_types}")
    
    # 生成所有弧
    print("\n[ArcAssembly] 生成所有弧类型...")
    arc_data = assembly.generate_all_arcs(save_individual=True)
    
    # 合并所有弧
    print("\n[ArcAssembly] 合并所有弧...")
    combined_df = assembly.combine_all_arcs(arc_data, save_combined=True)
    
    # 获取统计信息
    stats = assembly.get_statistics(arc_data)
    print(f"\n[ArcAssembly] 统计信息:")
    print(f"  总弧数量: {stats['total_arcs']}")
    print(f"  弧类型分布: {stats['arc_type_counts']}")
    print(f"  时间范围: {stats['time_range']}")
    
    # 保存元数据
    assembly.save_metadata(arc_data)
    
    print(f"\n[ArcAssembly] 完成！")
    print(f"  合并弧文件: data/intermediate/all_arcs_new.parquet")
    print(f"  元数据文件: data/intermediate/arcs_metadata_new.json")
    
    # 显示一些示例弧
    if not combined_df.empty:
        print(f"\n[ArcAssembly] 弧类型分布:")
        print(combined_df['arc_type'].value_counts())
        
        print(f"\n[ArcAssembly] 前5条弧:")
        print(combined_df.head())


if __name__ == "__main__":
    main()
