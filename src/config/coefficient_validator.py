#!/usr/bin/env python3
"""
系数验证和调整工具
用于确保模型参数满足无套利约束，防止空转获利和错配激励
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CoefficientConstraints:
    """系数约束配置 - 匹配当前模型结构"""
    # 基础成本系数（来自 network_config.py）
    gamma_rep: float  # 重定位时间成本系数（对应 gamma_rep_p）
    beta_toCHG: float  # 去充电站时间成本系数（对应 beta_chg_p1）
    beta_chg: float  # 充电占用成本系数（对应 beta_chg_p2）
    vot: float  # 时间价值
    
    # 奖励系数（来自 network_config.py）
    gamma_reposition_reward: float  # 重定位奖励系数（对应 gamma_rep_a）
    beta_chg_reward: float  # 充电奖励系数（对应 beta_chg_a）
    
    # 服务相关系数
    unmet_weight_default: float  # 未满足需求惩罚权重（对应 alpha_unmet）
    idle_opportunity_cost: float  # idle弧机会成本
    
    # 统计量（从实际数据计算）
    tt_rep_min: float  # 重定位时间最小值
    tt_rep_p50: float  # 重定位时间中位数
    tt_rep_p90: float  # 重定位时间90分位数
    tt_tochg_min: float  # 去站时间最小值
    delta_min_chg: float = 20.0  # SOC最小上调步长（更接近实际充电场景）
    
    # 充电相关参数
    charge_rate_min_per_soc: float = 1.0  # 每分钟充电SOC百分比
    de_tochg_km: float = 0.1  # 去充电站能耗系数
    dt_minutes: float = 15.0  # 时间步长度（分钟）
    
    # 调整参数
    epsilon: float = 0.01  # 安全边距
    eta: float = 1.0  # 词典序层级参数（分钟）

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    violations: List[str]
    adjustments: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]

class CoefficientValidator:
    """系数验证器"""
    
    def __init__(self, constraints: CoefficientConstraints):
        self.constraints = constraints
        self.violations = []
        self.adjustments = {}
        self.warnings = []
        self.recommendations = []
    
    def _calculate_net_coefficients(self) -> Dict[str, float]:
        """计算净系数"""
        return {
            'P': self.constraints.vot * self.constraints.unmet_weight_default,
            'beta_2_minus_alpha_chg': self.constraints.beta_chg - self.constraints.beta_chg_reward,
            'gamma_rep_scaled': self.constraints.gamma_rep * self.constraints.tt_rep_p50
        }
    
    def _minutes_to_steps(self, minutes: float) -> int:
        """将分钟转换为时间步（上取整）"""
        return int(np.ceil(minutes / self.constraints.dt_minutes))
    
    def _calculate_charging_time_steps(self, l_start: float, l_end: float, travel_time: float) -> int:
        """计算充电时间步数
        
        Args:
            l_start: 去站前电量（SOC百分比）
            l_end: 目标电量（SOC百分比）
            travel_time: 去站时间（分钟）
        
        Returns:
            充电时间步数
        """
        # 到站后的SOC（去站前电量 - 去站途中耗电量）
        arrived_soc = max(0, l_start - self.constraints.de_tochg_km * travel_time)
        # 需要充电的SOC量（目标电量 - 到站电量）
        soc_to_charge = l_end - arrived_soc
        if soc_to_charge <= 0:
            return 0
        # 充电时间（分钟）
        charging_minutes = soc_to_charge * self.constraints.charge_rate_min_per_soc
        # 转换为时间步
        return self._minutes_to_steps(charging_minutes)
    
    def _calculate_charging_reward(self, soc_change: float) -> float:
        """计算充电奖励
        
        Args:
            soc_change: SOC变化量（百分比，0-100）
        
        Returns:
            充电奖励（负值，表示收益）
        """
        return -self.constraints.beta_chg_reward * soc_change
    
    def _check_charging_no_free_lunch(self) -> bool:
        """检查约束A1: 充电本身不得成为净奖励"""
        net_coeff = self.constraints.beta_chg - self.constraints.beta_chg_reward
        if net_coeff < self.constraints.epsilon:
            self.violations.append(
                f"A1违反: β_chg - β_chg_reward = {net_coeff:.4f} < ε = {self.constraints.epsilon:.4f}"
            )
            # 自动调整
            self.adjustments['beta_chg_reward'] = self.constraints.beta_chg - self.constraints.epsilon
            self.recommendations.append(
                f"建议调整 β_chg_reward = {self.adjustments['beta_chg_reward']:.4f}"
            )
            return False
        return True
    
    def _check_reposition_recharge_no_arbitrage(self) -> bool:
        """检查约束A2: 重定位+就地补回能量不能净赚"""
        # 使用最保守的情况：最小重定位时间 + 最小充电增量
        # 计算最小充电时间步数
        min_charging_steps = self._calculate_charging_time_steps(
            self.constraints.delta_min_chg, 
            self.constraints.delta_min_chg + 1.0,  # 最小SOC增量
            self.constraints.tt_tochg_min
        )
        
        # 计算重定位时间步数
        reposition_steps = self._minutes_to_steps(self.constraints.tt_rep_min)
        # 计算去站时间步数
        tochg_steps = self._minutes_to_steps(self.constraints.tt_tochg_min)
        
        # 计算实际的SOC变化量（更现实的充电场景）
        realistic_soc_change = 30.0  # 从低电量充到中等电量，30%变化
        charging_reward = self._calculate_charging_reward(realistic_soc_change)
        
        min_cost = (self.constraints.gamma_rep * reposition_steps 
                   - self.constraints.gamma_reposition_reward * 1.0  # 区域价值不需要时间步转换
                   + self.constraints.beta_toCHG * tochg_steps
                   + self.constraints.beta_chg * min_charging_steps
                   + charging_reward)  # 使用计算出的充电奖励
        
        if min_cost < 0:
            self.violations.append(
                f"A2违反: 重定位+充电最小成本 = {min_cost:.4f} < 0"
            )
            # 建议调整策略
            self.recommendations.append("建议减小 γ_rep 或增大 β₂ - α_chg")
            return False
        return True
    
    def _check_serve_vs_charge(self) -> bool:
        """检查约束A3: 去站充电不应比直接服务更划算"""
        P = self.constraints.vot * self.constraints.unmet_weight_default
        
        # 计算最小充电时间步数
        min_charging_steps = self._calculate_charging_time_steps(
            self.constraints.delta_min_chg, 
            self.constraints.delta_min_chg + 1.0,  # 最小SOC增量
            self.constraints.tt_tochg_min
        )
        
        # 计算去站时间步数
        tochg_steps = self._minutes_to_steps(self.constraints.tt_tochg_min)
        
        # 使用更现实的SOC变化量计算充电奖励
        realistic_soc_change = 30.0
        charging_reward = self._calculate_charging_reward(realistic_soc_change)
        
        min_charge_cost = (self.constraints.beta_toCHG * tochg_steps 
                          + self.constraints.beta_chg * min_charging_steps
                          + charging_reward)
        
        if P < min_charge_cost:
            self.violations.append(
                f"A3违反: P = {P:.4f} < 最小充电成本 = {min_charge_cost:.4f}"
            )
            # 建议调整
            required_P = min_charge_cost + self.constraints.epsilon
            self.adjustments['unmet_weight_default'] = required_P / self.constraints.vot
            self.recommendations.append(
                f"建议调整 unmet_weight_default = {self.adjustments['unmet_weight_default']:.4f}"
            )
            return False
        return True
    
    def _check_serve_vs_reposition(self) -> bool:
        """检查约束A4: 去重定位不应比直接服务更划算"""
        P = self.constraints.vot * self.constraints.unmet_weight_default
        # 计算重定位时间步数
        reposition_steps = self._minutes_to_steps(self.constraints.tt_rep_min)
        
        max_reposition_benefit = (self.constraints.gamma_rep * reposition_steps 
                                 - self.constraints.gamma_reposition_reward * 1.0)  # 区域价值不需要时间步转换
        
        if P < max_reposition_benefit:
            self.violations.append(
                f"A4违反: P = {P:.4f} < 最大重定位收益 = {max_reposition_benefit:.4f}"
            )
            # 建议调整
            required_P = max_reposition_benefit + self.constraints.epsilon
            self.adjustments['unmet_weight_default'] = required_P / self.constraints.vot
            self.recommendations.append(
                f"建议调整 unmet_weight_default = {self.adjustments['unmet_weight_default']:.4f}"
            )
            return False
        return True
    
    def _check_lexicographic_order(self) -> bool:
        """检查约束B1: 词典序层级"""
        P = self.constraints.vot * self.constraints.unmet_weight_default
        
        # 计算重定位时间步数
        reposition_steps_p90 = self._minutes_to_steps(self.constraints.tt_rep_p90)
        tochg_steps = self._minutes_to_steps(self.constraints.tt_tochg_min)
        
        # 最大重定位净成本
        max_rep_cost = self.constraints.gamma_rep * reposition_steps_p90 - self.constraints.gamma_reposition_reward * 1.0  # 区域价值不需要时间步转换
        
        # 最大充电成本
        max_charging_steps = self._calculate_charging_time_steps(
            self.constraints.delta_min_chg, 
            self.constraints.delta_min_chg + 1.0,  # 最小SOC增量
            self.constraints.tt_tochg_min
        )
        max_chg_cost = (self.constraints.beta_toCHG * tochg_steps 
                       + self.constraints.beta_chg * max_charging_steps
                       - self.constraints.beta_chg_reward * self.constraints.delta_min_chg)
        
        violations = []
        if P < max_rep_cost + self.constraints.eta:
            violations.append(f"B1a: P = {P:.4f} < 最大重定位成本 + η = {max_rep_cost + self.constraints.eta:.4f}")
        
        if P < max_chg_cost + self.constraints.eta:
            violations.append(f"B1b: P = {P:.4f} < 最大充电成本 + η = {max_chg_cost + self.constraints.eta:.4f}")
        
        if violations:
            self.warnings.extend(violations)
            # 建议调整
            required_P = max(max_rep_cost, max_chg_cost) + self.constraints.eta + self.constraints.epsilon
            self.adjustments['unmet_weight_default'] = required_P / self.constraints.vot
            self.recommendations.append(
                f"建议调整 unmet_weight_default = {self.adjustments['unmet_weight_default']:.4f} 以确保词典序"
            )
            return False
        return True
    
    def _check_reposition_scale(self) -> bool:
        """检查约束B2: 限制纯重定位为负成本边的规模"""
        # 计算重定位时间步数
        reposition_steps_p50 = self._minutes_to_steps(self.constraints.tt_rep_p50)
        
        if self.constraints.gamma_reposition_reward * 1.0 > self.constraints.gamma_rep * reposition_steps_p50:
            self.warnings.append(
                f"B2警告: γ_reposition_reward×V_min = {self.constraints.gamma_reposition_reward * 1.0:.4f} > γ_rep×goStep_rep_p50 = {self.constraints.gamma_rep * reposition_steps_p50:.4f}"
            )
            self.adjustments['gamma_reposition_reward'] = (self.constraints.gamma_rep * reposition_steps_p50) / 1.0
            self.recommendations.append(
                f"建议调整 γ_reposition_reward = {self.adjustments['gamma_reposition_reward']:.4f}"
            )
            return False
        return True
    
    def _check_unit_consistency(self) -> bool:
        """检查约束B3: 统一单位的分钟等价范围"""
        # 计算重定位时间步数
        reposition_steps_p90 = self._minutes_to_steps(self.constraints.tt_rep_p90)
        
        scale_max = self.constraints.gamma_rep * reposition_steps_p90
        
        if self.constraints.gamma_reposition_reward * 1.0 > scale_max:
            self.warnings.append(
                f"B3警告: γ_reposition_reward×V_min = {self.constraints.gamma_reposition_reward * 1.0:.4f} > γ_rep×goStep_rep_p90 = {scale_max:.4f}"
            )
            self.adjustments['gamma_reposition_reward'] = scale_max / 1.0
            self.recommendations.append(
                f"建议调整 γ_reposition_reward = {self.adjustments['gamma_reposition_reward']:.4f} 以保持量纲一致"
            )
            return False
        return True
    
    def _check_idle_opportunity_cost(self) -> bool:
        """检查约束C1: idle弧机会成本合理性"""
        P = self.constraints.vot * self.constraints.unmet_weight_default
        
        # idle机会成本不应超过服务奖励
        if self.constraints.idle_opportunity_cost > P:
            self.warnings.append(
                f"C1警告: idle机会成本 = {self.constraints.idle_opportunity_cost:.4f} > 服务奖励P = {P:.4f}"
            )
            self.adjustments['idle_opportunity_cost'] = P * 0.8  # 设置为服务奖励的80%
            self.recommendations.append(
                f"建议调整 idle_opportunity_cost = {self.adjustments['idle_opportunity_cost']:.4f}"
            )
            return False
        return True
    
    def _check_charging_vs_reposition(self) -> bool:
        """检查约束C2: 充电与重定位的相对成本合理性"""
        # 计算最小充电成本
        min_charging_steps = self._calculate_charging_time_steps(
            self.constraints.delta_min_chg, 
            self.constraints.delta_min_chg + 1.0,
            self.constraints.tt_tochg_min
        )
        tochg_steps = self._minutes_to_steps(self.constraints.tt_tochg_min)
        
        min_charge_cost = (self.constraints.beta_toCHG * tochg_steps 
                          + self.constraints.beta_chg * min_charging_steps
                          - self.constraints.beta_chg_reward * self.constraints.delta_min_chg)
        
        # 计算最小重定位成本
        reposition_steps = self._minutes_to_steps(self.constraints.tt_rep_min)
        min_rep_cost = (self.constraints.gamma_rep * reposition_steps 
                       - self.constraints.gamma_reposition_reward * 1.0)
        
        # 充电成本与重定位成本应该在同一量级
        cost_ratio = min_charge_cost / max(min_rep_cost, 0.1)  # 避免除零
        
        if cost_ratio > 10.0:  # 充电成本不应比重定位成本高太多
            self.warnings.append(
                f"C2警告: 充电/重定位成本比 = {cost_ratio:.2f} > 10，可能过于昂贵"
            )
            self.recommendations.append("建议检查充电成本系数设置")
            return False
        elif cost_ratio < 0.1:  # 充电成本不应比重定位成本低太多
            self.warnings.append(
                f"C2警告: 充电/重定位成本比 = {cost_ratio:.2f} < 0.1，可能过于便宜"
            )
            self.recommendations.append("建议检查充电奖励系数设置")
            return False
        return True
    
    def validate(self) -> ValidationResult:
        """执行完整的验证"""
        logger.info("开始系数验证...")
        
        # 重置结果
        self.violations = []
        self.adjustments = {}
        self.warnings = []
        self.recommendations = []
        
        # 执行所有检查
        checks = [
            ("A1: 充电无免费午餐", self._check_charging_no_free_lunch),
            ("A2: 重定位+充电无套利", self._check_reposition_recharge_no_arbitrage),
            ("A3: 服务vs充电", self._check_serve_vs_charge),
            ("A4: 服务vs重定位", self._check_serve_vs_reposition),
            ("B1: 词典序层级", self._check_lexicographic_order),
            ("B2: 重定位规模限制", self._check_reposition_scale),
            ("B3: 单位一致性", self._check_unit_consistency),
            ("C1: idle机会成本", self._check_idle_opportunity_cost),
            ("C2: 充电vs重定位成本比", self._check_charging_vs_reposition),
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                logger.info(f"{check_name}: {'通过' if result else '违反'}")
            except Exception as e:
                logger.error(f"{check_name} 检查出错: {e}")
                self.violations.append(f"{check_name}: 检查出错 - {e}")
        
        # 生成建议
        if not self.violations and not self.warnings:
            self.recommendations.append("所有约束均满足，参数设置良好")
        
        is_valid = len(self.violations) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            violations=self.violations,
            adjustments=self.adjustments,
            warnings=self.warnings,
            recommendations=self.recommendations
        )
    
    def apply_adjustments(self, constraints: CoefficientConstraints) -> CoefficientConstraints:
        """应用调整建议"""
        if not self.adjustments:
            return constraints
        
        # 创建新的约束对象
        new_constraints = CoefficientConstraints(
            gamma_rep=constraints.gamma_rep,
            beta_toCHG=constraints.beta_toCHG,
            beta_chg=constraints.beta_chg,
            vot=constraints.vot,
            gamma_reposition_reward=self.adjustments.get('gamma_reposition_reward', constraints.gamma_reposition_reward),
            beta_chg_reward=self.adjustments.get('beta_chg_reward', constraints.beta_chg_reward),
            unmet_weight_default=self.adjustments.get('unmet_weight_default', constraints.unmet_weight_default),
            idle_opportunity_cost=constraints.idle_opportunity_cost,
            tt_rep_min=constraints.tt_rep_min,
            tt_rep_p50=constraints.tt_rep_p50,
            tt_rep_p90=constraints.tt_rep_p90,
            tt_tochg_min=constraints.tt_tochg_min,
            delta_min_chg=constraints.delta_min_chg,
            charge_rate_min_per_soc=constraints.charge_rate_min_per_soc,
            de_tochg_km=constraints.de_tochg_km,
            dt_minutes=constraints.dt_minutes,
            epsilon=constraints.epsilon,
            eta=constraints.eta
        )
        
        return new_constraints

def load_constraints_from_network_config() -> CoefficientConstraints:
    """从网络配置加载约束"""
    try:
        from network_config import get_network_config
        cfg = get_network_config()
        
        return CoefficientConstraints(
            gamma_rep=cfg.costs_equity.gamma_rep,
            beta_toCHG=cfg.costs_equity.beta_toCHG,
            beta_chg=cfg.costs_equity.beta_chg,
            vot=cfg.costs_equity.vot,
            gamma_reposition_reward=cfg.costs_equity.gamma_reposition_reward,
            beta_chg_reward=cfg.costs_equity.beta_chg_reward,
            unmet_weight_default=cfg.costs_equity.unmet_weight_default,
            idle_opportunity_cost=cfg.costs_equity.idle_opportunity_cost,
            dt_minutes=cfg.time_soc.dt_minutes,
            tt_rep_min=1.0,  # 将从数据计算
            tt_rep_p50=5.0,  # 将从数据计算
            tt_rep_p90=10.0,  # 将从数据计算
            tt_tochg_min=2.0,  # 将从数据计算
            delta_min_chg=cfg.charge_queue.min_charge_step,
            epsilon=0.01,
            eta=1.0
        )
    except ImportError:
        raise ImportError("无法导入网络配置模块，请确保在正确的目录中运行")
    except Exception as e:
        raise ValueError(f"从网络配置加载约束失败: {e}")

def load_constraints_from_config(config_path: str) -> CoefficientConstraints:
    """从JSON配置文件加载约束（向后兼容）"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 从配置中提取系数
    coefficients = config.get('coefficients', {})
    
    return CoefficientConstraints(
        gamma_rep=coefficients.get('gamma_rep', 1.0),
        beta_toCHG=coefficients.get('beta_toCHG', 1.0),
        beta_chg=coefficients.get('beta_chg', 1.0),
        vot=coefficients.get('vot', 1.0),
        gamma_reposition_reward=coefficients.get('gamma_reposition_reward', 0.5),
        beta_chg_reward=coefficients.get('beta_chg_reward', 0.5),
        unmet_weight_default=coefficients.get('unmet_weight_default', 10.0),
        idle_opportunity_cost=coefficients.get('idle_opportunity_cost', 10.0),
        tt_rep_min=coefficients.get('tt_rep_min', 1.0),
        tt_rep_p50=coefficients.get('tt_rep_p50', 5.0),
        tt_rep_p90=coefficients.get('tt_rep_p90', 10.0),
        tt_tochg_min=coefficients.get('tt_tochg_min', 2.0),
        delta_min_chg=coefficients.get('delta_min_chg', 15.0),
        charge_rate_min_per_soc=coefficients.get('charge_rate_min_per_soc', 1.0),
        de_tochg_km=coefficients.get('de_tochg_km', 0.1),
        dt_minutes=coefficients.get('dt_minutes', 15.0),
        epsilon=coefficients.get('epsilon', 0.01),
        eta=coefficients.get('eta', 1.0)
    )

def calculate_statistics_from_data(data_dir: str = "data") -> Dict[str, float]:
    """从实际数据文件计算统计量"""
    logger.info(f"从 {data_dir} 计算统计量...")
    
    stats = {}
    
    try:
        # 1. 从 base_ij 计算重定位时间统计量
        base_ij_path = Path(data_dir) / "base_ij.parquet"
        if base_ij_path.exists():
            base_ij = pd.read_parquet(base_ij_path)
            if 'base_minutes' in base_ij.columns:
                rep_times = base_ij['base_minutes']
                stats['tt_rep_min'] = float(rep_times.min())
                stats['tt_rep_p50'] = float(rep_times.median())
                stats['tt_rep_p90'] = float(rep_times.quantile(0.9))
                logger.info(f"重定位时间统计: min={stats['tt_rep_min']:.1f}, p50={stats['tt_rep_p50']:.1f}, p90={stats['tt_rep_p90']:.1f}")
        
        # 2. 从 base_i2k 计算去充电站时间统计量
        base_i2k_path = Path(data_dir) / "base_i2k.parquet"
        if base_i2k_path.exists():
            base_i2k = pd.read_parquet(base_i2k_path)
            if 'base_minutes' in base_i2k.columns:
                tochg_times = base_i2k['base_minutes']
                stats['tt_tochg_min'] = float(tochg_times.min())
                logger.info(f"去充电站时间统计: min={stats['tt_tochg_min']:.1f}")
        
        # 3. 从 reachability 数据计算更精确的统计量
        reachability_path = Path(data_dir) / "intermediate" / "reachability.parquet"
        if reachability_path.exists():
            reachability = pd.read_parquet(reachability_path)
            if 'travel_time_minutes' in reachability.columns:
                reach_times = reachability['travel_time_minutes']
                stats['tt_rep_min'] = min(stats.get('tt_rep_min', float('inf')), float(reach_times.min()))
                stats['tt_rep_p50'] = float(reach_times.median())
                stats['tt_rep_p90'] = float(reach_times.quantile(0.9))
                logger.info(f"可达性时间统计更新: min={stats['tt_rep_min']:.1f}, p50={stats['tt_rep_p50']:.1f}, p90={stats['tt_rep_p90']:.1f}")
        
    except Exception as e:
        logger.warning(f"计算统计量时出错: {e}")
    
    # 设置默认值
    stats.setdefault('tt_rep_min', 1.0)
    stats.setdefault('tt_rep_p50', 5.0)
    stats.setdefault('tt_rep_p90', 15.0)
    stats.setdefault('tt_tochg_min', 2.0)
    
    return stats

def generate_report(result: ValidationResult, output_path: str, constraints: CoefficientConstraints = None):
    """生成详细的验证报告"""
    report = {
        'validation_summary': {
            'is_valid': result.is_valid,
            'total_violations': len(result.violations),
            'total_warnings': len(result.warnings),
            'total_recommendations': len(result.recommendations),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        },
        'current_coefficients': {
            'gamma_rep': constraints.gamma_rep if constraints else None,
            'beta_toCHG': constraints.beta_toCHG if constraints else None,
            'beta_chg': constraints.beta_chg if constraints else None,
            'vot': constraints.vot if constraints else None,
            'gamma_reposition_reward': constraints.gamma_reposition_reward if constraints else None,
            'beta_chg_reward': constraints.beta_chg_reward if constraints else None,
            'unmet_weight_default': constraints.unmet_weight_default if constraints else None,
            'idle_opportunity_cost': constraints.idle_opportunity_cost if constraints else None,
        } if constraints else None,
        'violations': result.violations,
        'warnings': result.warnings,
        'adjustments': result.adjustments,
        'recommendations': result.recommendations,
        'arbitrage_analysis': {
            'service_vs_charging_arbitrage': any('A3' in v for v in result.violations),
            'service_vs_reposition_arbitrage': any('A4' in v for v in result.violations),
            'charging_free_lunch': any('A1' in v for v in result.violations),
            'reposition_charging_arbitrage': any('A2' in v for v in result.violations),
            'lexicographic_order_violation': any('B1' in v for v in result.violations),
            'scale_consistency_issues': any('B2' in v or 'B3' in v for v in result.violations),
            'idle_cost_issues': any('C1' in v for v in result.violations),
            'relative_cost_issues': any('C2' in v for v in result.violations),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"验证报告已保存到: {output_path}")
    
    # 生成简化的文本报告
    text_report_path = output_path.replace('.json', '_summary.txt')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("当前模型无套利成本-奖励系数检测报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"验证状态: {'✅ 通过' if result.is_valid else '❌ 失败'}\n")
        f.write(f"违反约束数: {len(result.violations)}\n")
        f.write(f"警告数: {len(result.warnings)}\n\n")
        
        if result.violations:
            f.write("🚨 违反的约束:\n")
            for violation in result.violations:
                f.write(f"  ❌ {violation}\n")
            f.write("\n")
        
        if result.warnings:
            f.write("⚠️ 警告:\n")
            for warning in result.warnings:
                f.write(f"  ⚠️  {warning}\n")
            f.write("\n")
        
        if result.adjustments:
            f.write("📝 建议调整:\n")
            for param, value in result.adjustments.items():
                f.write(f"  📝 {param} = {value:.4f}\n")
            f.write("\n")
        
        if result.recommendations:
            f.write("💡 其他建议:\n")
            for rec in result.recommendations:
                f.write(f"  💡 {rec}\n")
            f.write("\n")
        
        # 套利分析摘要
        f.write("🔍 套利分析摘要:\n")
        analysis = report['arbitrage_analysis']
        if analysis['service_vs_charging_arbitrage']:
            f.write("  ❌ 检测到服务vs充电套利机会\n")
        if analysis['service_vs_reposition_arbitrage']:
            f.write("  ❌ 检测到服务vs重定位套利机会\n")
        if analysis['charging_free_lunch']:
            f.write("  ❌ 检测到充电免费午餐问题\n")
        if analysis['reposition_charging_arbitrage']:
            f.write("  ❌ 检测到重定位+充电套利机会\n")
        
        if not any(analysis.values()):
            f.write("  ✅ 未检测到明显的套利机会\n")
        
        f.write("\n" + "="*60 + "\n")
    
    logger.info(f"简化报告已保存到: {text_report_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='当前模型无套利成本-奖励系数检测脚本')
    parser.add_argument('--config', help='JSON配置文件路径（可选，默认使用网络配置）')
    parser.add_argument('--data-dir', default='data', help='数据目录路径（用于计算统计量）')
    parser.add_argument('--output', default='validation_report.json', help='输出报告路径')
    parser.add_argument('--auto-adjust', action='store_true', help='自动应用调整建议')
    parser.add_argument('--use-network-config', action='store_true', default=True, help='使用网络配置（默认）')
    
    args = parser.parse_args()
    
    try:
        # 加载约束
        if args.config:
            constraints = load_constraints_from_config(args.config)
        else:
            constraints = load_constraints_from_network_config()
        
        # 计算统计量
        stats = calculate_statistics_from_data(args.data_dir)
        constraints.tt_rep_min = stats['tt_rep_min']
        constraints.tt_rep_p50 = stats['tt_rep_p50']
        constraints.tt_rep_p90 = stats['tt_rep_p90']
        constraints.tt_tochg_min = stats['tt_tochg_min']
        
        # 执行验证
        validator = CoefficientValidator(constraints)
        result = validator.validate()
        
        # 打印结果
        print("\n" + "="*60)
        print("系数验证结果")
        print("="*60)
        
        print(f"验证状态: {'通过' if result.is_valid else '失败'}")
        print(f"违反约束数: {len(result.violations)}")
        print(f"警告数: {len(result.warnings)}")
        
        if result.violations:
            print("\n违反的约束:")
            for violation in result.violations:
                print(f"  ❌ {violation}")
        
        if result.warnings:
            print("\n警告:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        if result.adjustments:
            print("\n建议调整:")
            for param, value in result.adjustments.items():
                print(f"  📝 {param} = {value:.4f}")
        
        if result.recommendations:
            print("\n其他建议:")
            for rec in result.recommendations:
                print(f"  💡 {rec}")
        
        # 生成报告
        generate_report(result, args.output, constraints)
        
        # 自动调整
        if args.auto_adjust and result.adjustments:
            adjusted_constraints = validator.apply_adjustments(constraints)
            
            if args.config:
                adjusted_config_path = args.config.replace('.json', '_adjusted.json')
                # 保存调整后的JSON配置
                adjusted_config = {
                    'coefficients': {
                        'gamma_rep': adjusted_constraints.gamma_rep,
                        'beta_toCHG': adjusted_constraints.beta_toCHG,
                        'beta_chg': adjusted_constraints.beta_chg,
                        'vot': adjusted_constraints.vot,
                        'gamma_reposition_reward': adjusted_constraints.gamma_reposition_reward,
                        'beta_chg_reward': adjusted_constraints.beta_chg_reward,
                        'unmet_weight_default': adjusted_constraints.unmet_weight_default,
                        'idle_opportunity_cost': adjusted_constraints.idle_opportunity_cost,
                        'epsilon': adjusted_constraints.epsilon,
                        'eta': adjusted_constraints.eta
                    }
                }
                with open(adjusted_config_path, 'w') as f:
                    json.dump(adjusted_config, f, indent=2)
                print(f"\n✅ 调整后的配置已保存到: {adjusted_config_path}")
            else:
                print(f"\n✅ 调整后的系数:")
                for param, value in result.adjustments.items():
                    print(f"  - {param} = {value:.4f}")
                print("\n💡 请手动更新 network_config.py 中的相应系数")
            
            # 重新验证调整后的配置
            print("\n重新验证调整后的配置...")
            new_validator = CoefficientValidator(adjusted_constraints)
            new_result = new_validator.validate()
            print(f"调整后验证状态: {'通过' if new_result.is_valid else '失败'}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        return 1
    
    return 0 if result.is_valid else 1

if __name__ == "__main__":
    exit(main())
