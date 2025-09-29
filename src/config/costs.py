# 06_costs.py
# 成本系数准备 & 求解后评估（适配 03/04/05 裁剪 + 滚动LP + 动态弧，兼容纯网络化的 svc_* / chg_*）
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from config.network_config import get_network_config as get_config


def _to_finite_float(s: pd.Series, default: float = 0.0) -> pd.Series:
    """
    把任意序列安全转为 float64：
      - 非数字/无法解析 → NaN
      - NaN/±inf → default（默认 0.0）
    """
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(np.isfinite(x), default)
    return x.astype(float)


def _get_flag(name: str, default: bool = True) -> bool:
    """安全读取 cfg.flags.{name}，缺失时用默认值。"""
    try:
        return bool(get_config().flags.__dict__.get(name, default))
    except Exception:
        return bool(default)


# =========================================
# 常量：细分类的 arc_type（与 04 改造后保持一致）
# =========================================
IDLE_TYPES        = {"idle"}
SERVICE_TYPES     = {"svc_enter", "svc_gate", "svc_exit"}
REPOSITION_TYPES  = {"reposition"}
CHARGING_TYPES    = {"tochg", "chg_enter", "chg_occ", "chg_step"}


# =========================
# 内置：系数调度器 CoeffProvider
# =========================
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


# -------------------------
# 基础 I/O
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _pick_arc_path(basename: str) -> Path:
    """
    优先使用裁剪后的弧（*.pruned.parquet），不存在则回退到全量。
    basename 示例: 'reposition_arcs' / 'charging_arcs' / 'service_arcs'
    """
    inter = Path("data/intermediate")
    pruned = inter / f"{basename}.pruned.parquet"
    full   = inter / f"{basename}.parquet"
    if pruned.exists(): return pruned
    return full


# ============================================================
# 基于净需求的重定位奖励计算
# ============================================================
def build_net_demand_based_reposition_rewards() -> pd.DataFrame:
    """
    基于净需求计算重定位奖励
    
    Returns:
        DataFrame: [t, j, zone_value, zone_value_raw]
        zone_value: 归一化的净需求强度 (0-1)
        zone_value_raw: 原始净需求值
    """
    cfg = get_config()
    
    # 读取 OD 需求数据
    from utils.data_loader import load_od_matrix
    od = load_od_matrix(cfg)
    
    # 计算每个区域在每个时刻的净需求
    zone_demand_data = []
    
    # 获取所有时间步和区域
    all_times = sorted(od['t'].unique())
    all_zones = sorted(od['i'].unique())
    
    for t in all_times:
        # 计算该时刻所有区域的净需求
        net_demands = []
        zone_net_data = {}
        
        for zone in all_zones:
            # 计算入站需求（作为到达区域）
            inbound_demand = od[(od["t"] == t) & (od["j"] == zone)]["demand"].sum()
            
            # 计算出站需求（作为出发区域）
            outbound_demand = od[(od["t"] == t) & (od["i"] == zone)]["demand"].sum()
            
            # 计算净需求
            net_demand = inbound_demand - outbound_demand
            
            zone_net_data[zone] = net_demand
            net_demands.append(net_demand)
        
        # 归一化净需求强度
        if net_demands:
            max_net_demand = max(net_demands)
            min_net_demand = min(net_demands)
            net_range = max_net_demand - min_net_demand
            
            for zone in all_zones:
                net_demand = zone_net_data[zone]
                
                if net_range > 0:
                    # 将净需求映射到0-1范围
                    zone_value = max(0.0, min(1.0, (net_demand - min_net_demand) / net_range))
                else:
                    # 所有区域净需求相同，设为0.5
                    zone_value = 0.5
                
                zone_demand_data.append({
                    "t": t,
                    "j": zone,
                    "zone_value": zone_value,
                    "zone_value_raw": net_demand
                })
    
    return pd.DataFrame(zone_demand_data)


# ============================================================
# 1) 重定位成本系数（静态 & 动态）
# ============================================================
def build_reposition_cost_coefficients(save: bool = True) -> pd.DataFrame:
    """
    Crep(a) = VOT × Σ_{t_start}^{t_start+tau-1} α(tp)
    输入：data/intermediate/reposition_arcs(.pruned).parquet
    输出：[arc_id?, from_node_id, to_node_id, t, coef_rep]
    """
    rep_path = _pick_arc_path("reposition_arcs")
    if not rep_path.exists():
        return pd.DataFrame(columns=["from_node_id", "to_node_id", "t", "coef_rep"])

    rep = pd.read_parquet(rep_path)
    need = {"from_node_id","to_node_id","t","tau"}
    if rep.empty or not need.issubset(rep.columns):
        return pd.DataFrame(columns=["from_node_id", "to_node_id", "t", "coef_rep"])

    cfg = get_config()
    cp = CoeffProvider(schedule_path=cfg.paths.coeff_schedule)

    rep = rep.copy()
    rep["t"] = rep["t"].astype(int)
    rep["tau"] = rep["tau"].astype(int)
    rep["coef_rep"] = rep.apply(lambda r: cp.vot * cp.gamma_rep_p_sum_over_window(int(r["t"]), int(r["tau"])), axis=1)

    out_cols = ["from_node_id", "to_node_id", "t", "coef_rep"]
    if "arc_id" in rep.columns: out_cols.insert(0, "arc_id")
    out = rep[out_cols].drop_duplicates(subset=[c for c in out_cols if c != "coef_rep"])
    if save:
        out.to_parquet("data/intermediate/reposition_costs.parquet", index=False)
    return out


def build_reposition_cost_coefficients_from_arcs(rep_arcs: pd.DataFrame, cp: Optional[CoeffProvider] = None) -> pd.DataFrame:
    """窗口版：对给定的重定位弧表计算 coef_rep。"""
    if rep_arcs is None or rep_arcs.empty:
        return pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","coef_rep"])
    need = {"from_node_id","to_node_id","t","tau"}
    if not need.issubset(rep_arcs.columns):
        return pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","coef_rep"])
    cfg = get_config()
    cp = cp or CoeffProvider(cfg.paths.coeff_schedule)
    rep = rep_arcs.copy()
    rep["t"] = rep["t"].astype(int)
    rep["tau"] = rep["tau"].astype(int)
    rep["coef_rep"] = rep.apply(lambda r: cp.vot * cp.gamma_rep_p_sum_over_window(int(r["t"]), int(r["tau"])), axis=1)
    out_cols = ["from_node_id","to_node_id","t","coef_rep"]
    if "arc_id" in rep.columns: out_cols = ["arc_id"] + out_cols
    return rep[out_cols].drop_duplicates(subset=[c for c in out_cols if c != "coef_rep"])


# ============================================================
# 2) 充电成本（tochg / chg_occ）
# ============================================================
def occ_notna_mask(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in cols:
        m &= df[c].notna()
    return m

def build_charging_cost_coefficients(save: bool = True) -> Dict[str, str]:
    """
    读取 data/intermediate/charging_arcs(.pruned).parquet，生成两张成本表：
      - charging_travel_costs.parquet:  (仅 'tochg')  列 [arc_id?, from_node_id, to_node_id, t, tau_tochg, coef_chg_travel]
      - charging_occupancy_costs.parquet: (仅 'chg_occ') 列 [arc_id?, from_node_id, to_node_id, t, coef_chg_occ]
    返回生成路径字典。
    """
    chg_path = _pick_arc_path("charging_arcs")
    if not chg_path.exists():
        return {"charging_travel_costs": "", "charging_occupancy_costs": ""}

    chg = pd.read_parquet(chg_path)
    if chg.empty:
        return {"charging_travel_costs": "", "charging_occupancy_costs": ""}

    cfg = get_config()
    cp = CoeffProvider(schedule_path=cfg.paths.coeff_schedule)

    out_dir = Path("data/intermediate")
    ensure_dir(out_dir)

    # 2.1 tochg：行驶成本
    cols_tr = ["from_node_id","to_node_id","t","tau_tochg"]
    if "arc_type" in chg.columns:
        tochg = chg.loc[chg["arc_type"].eq("tochg") & chg[cols_tr].notna().all(axis=1)].copy()
    else:
        tochg = pd.DataFrame(columns=cols_tr)
    if not tochg.empty:
        tochg["t"] = tochg["t"].astype(int)
        tochg["tau_tochg"] = tochg["tau_tochg"].astype(int).clip(lower=0)
        tochg["coef_chg_travel"] = tochg.apply(lambda r: cp.vot * cp.beta_chg_p1_sum_over_window(int(r["t"]), int(r["tau_tochg"])), axis=1)
        out_cols_tr = cols_tr + ["coef_chg_travel"]
        if "arc_id" in tochg.columns: out_cols_tr = ["arc_id"] + out_cols_tr
        tochg_out = tochg[out_cols_tr].drop_duplicates(subset=[c for c in out_cols_tr if c not in {"coef_chg_travel","tau_tochg"}])
        tochg_out.to_parquet(out_dir / "charging_travel_costs.parquet", index=False)
        p_tr = str(out_dir / "charging_travel_costs.parquet")
    else:
        p_tr = ""

    # 2.2 chg_occ：每步占用成本
    cols_occ = ["from_node_id","to_node_id","t"]
    if "arc_type" in chg.columns:
        occ = chg.loc[chg["arc_type"].eq("chg_occ") & occ_notna_mask(chg, cols_occ)].copy()
    else:
        occ = pd.DataFrame(columns=cols_occ)
    if not occ.empty:
        occ["t"] = occ["t"].astype(int)
        occ["coef_chg_occ"] = occ["t"].apply(lambda p: cp.vot * cp.beta_chg_p2(int(p)))
        out_cols_occ = cols_occ + ["coef_chg_occ"]
        if "arc_id" in occ.columns: out_cols_occ = ["arc_id"] + out_cols_occ
        occ_out = occ[out_cols_occ].drop_duplicates(subset=[c for c in out_cols_occ if c != "coef_chg_occ"])
        occ_out.to_parquet(out_dir / "charging_occupancy_costs.parquet", index=False)
        p_occ = str(out_dir / "charging_occupancy_costs.parquet")
    else:
        p_occ = ""

    return {"charging_travel_costs": p_tr, "charging_occupancy_costs": p_occ}


def build_charging_travel_costs_from_arcs(tochg_arcs: pd.DataFrame, cp: Optional[CoeffProvider] = None) -> pd.DataFrame:
    """窗口版：对 'tochg' 弧构建行驶成本。"""
    if tochg_arcs is None or tochg_arcs.empty:
        return pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","tau_tochg","coef_chg_travel"])
    need = {"from_node_id","to_node_id","t","tau_tochg"}
    if not need.issubset(tochg_arcs.columns):
        return pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","tau_tochg","coef_chg_travel"])
    cfg = get_config()
    cp = cp or CoeffProvider(cfg.paths.coeff_schedule)
    df = tochg_arcs.copy()
    df["t"] = df["t"].astype(int)
    df["tau_tochg"] = df["tau_tochg"].astype(int).clip(lower=0)
    df["coef_chg_travel"] = df.apply(lambda r: cp.vot * cp.beta_chg_p1_sum_over_window(int(r["t"]), int(r["tau_tochg"])), axis=1)
    out_cols = ["from_node_id","to_node_id","t","tau_tochg","coef_chg_travel"]
    if "arc_id" in df.columns: out_cols = ["arc_id"] + out_cols
    return df[out_cols].drop_duplicates(subset=[c for c in out_cols if c not in {"coef_chg_travel","tau_tochg"}])


def build_charging_occupancy_costs_from_arcs(occ_arcs: pd.DataFrame, cp: Optional[CoeffProvider] = None) -> pd.DataFrame:
    """窗口版：对 'chg_occ' 弧构建每步占用成本（按 t 点取 β2 + tou_price）。"""
    if occ_arcs is None or occ_arcs.empty:
        return pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","coef_chg_occ"])
    need = {"from_node_id","to_node_id","t"}
    if not need.issubset(occ_arcs.columns):
        return pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","coef_chg_occ"])
    cfg = get_config()
    cp = cp or CoeffProvider(cfg.paths.coeff_schedule)
    df = occ_arcs.copy()
    df["t"] = df["t"].astype(int)
    # 充电占用成本 = VOT * β_chg_p2(t) + tou_price(t)
    df["coef_chg_occ"] = df["t"].apply(lambda p: cp.vot * cp.beta_chg_p2(int(p)) + cp.tou_price(int(p)))
    out_cols = ["from_node_id","to_node_id","t","coef_chg_occ"]
    if "arc_id" in df.columns: out_cols = ["arc_id"] + out_cols
    return df[out_cols].drop_duplicates(subset=[c for c in out_cols if c != "coef_chg_occ"])


# ============================================================
# 3) 服务奖励（用于 svc_gate 负成本；等价替代“未满足惩罚”）
# ============================================================
def build_service_reward_coefficients(save: bool = True) -> pd.DataFrame:
    """
    服务奖励系数（用于给 svc_gate 弧施加负成本）：
      默认使用与“未满足惩罚”相同的权重：coef_svc_reward = VOT * unmet_weight
      可继续通过 cfg.unmet_weights_overrides 定义 (t, (i,j)->w) 的覆盖。
    输出：DataFrame [t, i, j, coef_svc_reward]
    """
    cfg = get_config()
    from utils.data_loader import load_od_matrix
    od = load_od_matrix(cfg)

    # 若希望与单期一致，可把 service_weight_default
    base = float(cfg.costs_equity.service_weight_default)
    vot = float(cfg.costs_equity.vot)

    od = od.copy()
    od["coef_svc_reward"] = vot * base

    # 复用 overrides（若存在）
    if hasattr(cfg, "unmet_weights_overrides") and cfg.unmet_weights_overrides:
        for t_key, pair_weights in cfg.unmet_weights_overrides.items():
            if isinstance(pair_weights, dict):
                for ij_key, w in pair_weights.items():
                    if isinstance(ij_key, tuple) and len(ij_key) == 2:
                        i_val, j_val = ij_key
                    elif isinstance(ij_key, str) and ij_key.startswith("("):
                        i_str, j_str = ij_key.strip("()").split(",")
                        i_val, j_val = int(i_str), int(j_str)
                    else:
                        continue
                    mask = (od["t"] == int(t_key)) & (od["i"] == int(i_val)) & (od["j"] == int(j_val))
                    od.loc[mask, "coef_svc_reward"] = vot * float(w)

    out = od.loc[:, ["t", "i", "j", "coef_svc_reward"]].drop_duplicates().reset_index(drop=True)
    if save:
        out.to_parquet("data/intermediate/service_gate_rewards.parquet", index=False)
    return out


# 兼容：保留未满足惩罚（若仍需单独评估/报告）
def build_unmet_penalty_coefficients(save: bool = True) -> pd.DataFrame:
    """
    未满足需求系数：
      coef_unmet = VOT * w_{i,j,t}
    输出：DataFrame [t, i, j, coef_unmet]
    """
    cfg = get_config()
    from utils.data_loader import load_od_matrix
    od = load_od_matrix(cfg)

    base = float(cfg.costs_equity.service_weight_default)
    vot = float(cfg.costs_equity.vot)

    od = od.copy()
    od["coef_unmet"] = vot * base

    if hasattr(cfg, "unmet_weights_overrides") and cfg.unmet_weights_overrides:
        for t_key, pair_weights in cfg.unmet_weights_overrides.items():
            if isinstance(pair_weights, dict):
                for ij_key, w in pair_weights.items():
                    if isinstance(ij_key, tuple) and len(ij_key) == 2:
                        i_val, j_val = ij_key
                    elif isinstance(ij_key, str) and ij_key.startswith("("):
                        i_str, j_str = ij_key.strip("()").split(",")
                        i_val, j_val = int(i_str), int(j_str)
                    else:
                        continue
                    mask = (od["t"] == int(t_key)) & (od["i"] == int(i_val)) & (od["j"] == int(j_val))
                    od.loc[mask, "coef_unmet"] = vot * float(w)

    out = od.loc[:, ["t", "i", "j", "coef_unmet"]].drop_duplicates().reset_index(drop=True)
    if save:
        out.to_parquet("data/intermediate/unmet_penalty.parquet", index=False)
    return out


# ============================================================
# 4) 窗口：一次性为弧表附加成本（含“收益”）
# ============================================================
def _merge_with_keys(flow_df: pd.DataFrame, coef_df: pd.DataFrame, keys: List[str], coef_col: str) -> pd.DataFrame:
    """
    安全合并：只携带 join keys + coef_col，避免把无关同名列带入导致 *_x/*_y 冲突。
    优先用 arc_id；否则退回到传入 keys 里双方都存在的键。
    合并后将 coef_col 统一净化为有限 float（NaN/±inf → 0.0）。
    当 flow_df 已有 coef_col 时，**新值优先覆盖旧值**。
    """
    if flow_df is None or flow_df.empty or coef_df is None or coef_df.empty:
        out = flow_df.copy()
        if coef_col not in out.columns:
            out[coef_col] = 0.0
        out[coef_col] = _to_finite_float(out[coef_col], 0.0)
        return out

    # 1) 选择 join 键
    if "arc_id" in flow_df.columns and "arc_id" in coef_df.columns:
        use_keys = ["arc_id"]
    else:
        use_keys = [k for k in keys if (k in flow_df.columns and k in coef_df.columns)]
        if not use_keys:
            out = flow_df.copy()
            if coef_col not in out.columns:
                out[coef_col] = 0.0
            out[coef_col] = _to_finite_float(out[coef_col], 0.0)
            return out

    # 2) 仅保留 keys + coef_col
    coef_slim_cols = list(dict.fromkeys(use_keys + [coef_col]))
    coef_slim = coef_df[coef_slim_cols].copy()
    coef_slim[coef_col] = _to_finite_float(coef_slim[coef_col], 0.0)

    # 3) 若 flow 已含 coef_col：用临时名回填（新值优先）
    out = flow_df.copy()
    if coef_col in out.columns:
        tmp = f"__{coef_col}_new"
        merged = out.merge(coef_slim.rename(columns={coef_col: tmp}), on=use_keys, how="left")
        merged[tmp] = _to_finite_float(merged.get(tmp, np.nan), 0.0)
        merged[coef_col] = merged[tmp].combine_first(pd.to_numeric(merged[coef_col], errors="coerce"))
        merged[coef_col] = _to_finite_float(merged[coef_col], 0.0)
        return merged.drop(columns=[tmp], errors="ignore")

    # 4) 正常合并
    out = out.merge(coef_slim, on=use_keys, how="left")
    out[coef_col] = _to_finite_float(out[coef_col], 0.0)
    return out


def build_cost_coefficients_for_window(arc_df_win: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    给窗口弧表即时生成成本系数与收益系数。
    返回 dict：
      {
        "rep":         df[arc_id?, from_node_id, to_node_id, t, coef_rep],
        "tochg":       df[arc_id?, from_node_id, to_node_id, t, tau_tochg, coef_chg_travel],
        "occ":         df[arc_id?, from_node_id, to_node_id, t, coef_chg_occ],
        "svc":         df[arc_id?, from_node_id, to_node_id, t, i, j, coef_svc_gate],  # 负成本
        "rep_reward":  df[arc_id?, t, coef_rep_reward],   # 负成本：重定位收益
        "chg_reward":  df[arc_id?, t, coef_chg_reward],   # 负成本：充电收益（按 ΔSOC）
      }
    """
    if arc_df_win is None or arc_df_win.empty:
        return {
            "rep": pd.DataFrame(), "tochg": pd.DataFrame(), "occ": pd.DataFrame(), "svc": pd.DataFrame(),
            "rep_reward": pd.DataFrame(), "chg_reward": pd.DataFrame()
        }

    cfg = get_config()
    cp = CoeffProvider(cfg.paths.coeff_schedule)

    # ---------- 成本：原有三类 ----------
    # 重定位
    rep_arcs = arc_df_win[arc_df_win["arc_type"].isin(REPOSITION_TYPES)].copy()
    rep_coef = build_reposition_cost_coefficients_from_arcs(rep_arcs, cp=cp)
    if not rep_coef.empty:
        rep_coef["coef_rep"] = _to_finite_float(rep_coef["coef_rep"], 0.0)

    # 去充电行驶
    tochg_arcs = arc_df_win[arc_df_win["arc_type"].eq("tochg")].copy()
    tochg_coef = build_charging_travel_costs_from_arcs(tochg_arcs, cp=cp)
    if not tochg_coef.empty:
        tochg_coef["coef_chg_travel"] = _to_finite_float(tochg_coef["coef_chg_travel"], 0.0)

    # 充电占用（逐步）
    occ_arcs = arc_df_win[arc_df_win["arc_type"].eq("chg_occ")].copy()
    occ_coef = build_charging_occupancy_costs_from_arcs(occ_arcs, cp=cp)
    if not occ_coef.empty:
        occ_coef["coef_chg_occ"] = _to_finite_float(occ_coef["coef_chg_occ"], 0.0)

    # 服务奖励（负成本）
    svc_coef = pd.DataFrame(columns=["arc_id","from_node_id","to_node_id","t","i","j","coef_svc_gate"])
    if _get_flag("enable_service_reward", True):
        svc_gate = arc_df_win[arc_df_win["arc_type"].eq("svc_gate")][["arc_id","from_node_id","to_node_id","t","i","j"]].copy()
        if not svc_gate.empty:
            rewards = build_service_reward_coefficients(save=False)
            svc = svc_gate.merge(rewards, on=["t","i","j"], how="left")
            svc["coef_svc_gate"] = -_to_finite_float(svc["coef_svc_reward"], 0.0)
            svc_coef = svc[["arc_id","from_node_id","to_node_id","t","i","j","coef_svc_gate"]]

    # ---------- 新增收益：重定位收益 & 充电收益 ----------
    rep_reward_coef = pd.DataFrame(columns=["arc_id","t","coef_rep_reward"])
    if _get_flag("enable_reposition_reward", True):
        if not rep_arcs.empty and "j" in rep_arcs.columns:
            zone_val = build_net_demand_based_reposition_rewards()  # [t, j, zone_value]
            gamma_rep_a = float(cfg.costs_equity.gamma_reposition_reward)
            tmp = rep_arcs[["arc_id","t","j"]].copy()
            tmp["t"] = tmp["t"].astype(int)
            rep_reward = tmp.merge(zone_val, on=["t","j"], how="left")
            rep_reward["zone_value"] = _to_finite_float(rep_reward["zone_value"], 0.0)
            rep_reward["coef_rep_reward"] = - float(gamma_rep_a) * rep_reward["zone_value"]
            rep_reward_coef = rep_reward[["arc_id","t","coef_rep_reward"]].drop_duplicates()

    chg_reward_coef = pd.DataFrame(columns=["arc_id","t","coef_chg_reward"])
    if _get_flag("enable_charging_reward", True):
        beta_chg_a = float(cfg.costs_equity.beta_chg_reward)
        # 优先使用 chg_step（特别是最后一步），因为只有chg_step有SOC变化
        src = arc_df_win.loc[arc_df_win["arc_type"].eq("chg_step"), ["arc_id","t","l","l_to","is_last_step"]].copy()
        if not src.empty and {"l","l_to"}.issubset(src.columns):
            src["t"] = src["t"].astype(int)
            src["d_soc"] = (src["l_to"].astype(int) - src["l"].astype(int)).clip(lower=0)
            # 只对最后一步的充电给予奖励，或者对所有有SOC提升的步骤给予奖励
            src["coef_chg_reward"] = - float(beta_chg_a) * src["d_soc"].astype(float)
            chg_reward_coef = src[["arc_id","t","coef_chg_reward"]].drop_duplicates()

    return {
        "rep": rep_coef, "tochg": tochg_coef, "occ": occ_coef, "svc": svc_coef,
        "rep_reward": rep_reward_coef, "chg_reward": chg_reward_coef
    }


def attach_costs_to_arcs_for_window(arc_df_win: pd.DataFrame) -> pd.DataFrame:
    """
    直接把成本/收益系数附加到窗口弧表（便于 07 在建模时直接读取）。
    输出列新增：
      - coef_rep：重定位弧成本（仅 reposition）
      - coef_chg_travel：去充电行驶成本（仅 tochg）
      - coef_chg_occ：每步充电占用成本（仅 chg_occ）
      - coef_svc_gate：服务“闸门弧”的负成本（仅 svc_gate）
      - coef_rep_reward：重定位收益（负成本；仅 reposition）
      - coef_chg_reward：充电收益（负成本；仅 chg_occ 或 chg_step）
      - coef_total：对每条弧的“净成本” = 上述各项相加（未涉及类型默认为 0）
    """
    if arc_df_win is None or arc_df_win.empty:
        return arc_df_win

    parts = build_cost_coefficients_for_window(arc_df_win)
    out = arc_df_win.copy()

    # 初始化（先置 NaN，让 merge 的新值可以覆盖）
    for c in ["coef_rep","coef_chg_travel","coef_chg_occ","coef_svc_gate",
              "coef_rep_reward","coef_chg_reward","coef_idle","coef_total"]:
        if c not in out.columns:
            out[c] = np.nan

    # 合并（按 arc_id 优先，内部会选 arc_id 或 keys）
    if not parts["rep"].empty:
        out = _merge_with_keys(out, parts["rep"], ["from_node_id","to_node_id","t"], "coef_rep")
    if not parts["tochg"].empty:
        out = _merge_with_keys(out, parts["tochg"], ["from_node_id","to_node_id","t"], "coef_chg_travel")
    if not parts["occ"].empty:
        out = _merge_with_keys(out, parts["occ"], ["from_node_id","to_node_id","t"], "coef_chg_occ")
    if not parts["svc"].empty:
        # svc_gate 用 arc_id+t 对齐（新值优先）
        out = out.merge(parts["svc"][["arc_id","t","coef_svc_gate"]],
                        on=["arc_id","t"], how="left", suffixes=("", "_new"))
        out["coef_svc_gate"] = out["coef_svc_gate_new"].combine_first(out["coef_svc_gate"])
        out = out.drop(columns=["coef_svc_gate_new"], errors="ignore")
    if not parts["rep_reward"].empty:
        out = out.merge(parts["rep_reward"][["arc_id","t","coef_rep_reward"]],
                        on=["arc_id","t"], how="left", suffixes=("", "_new"))
        out["coef_rep_reward"] = out["coef_rep_reward_new"].combine_first(out["coef_rep_reward"])
        out = out.drop(columns=["coef_rep_reward_new"], errors="ignore")
    if not parts["chg_reward"].empty:
        out = out.merge(parts["chg_reward"][["arc_id","t","coef_chg_reward"]],
                        on=["arc_id","t"], how="left", suffixes=("", "_new"))
        out["coef_chg_reward"] = out["coef_chg_reward_new"].combine_first(out["coef_chg_reward"])
        out = out.drop(columns=["coef_chg_reward_new"], errors="ignore")

    # 仅在对应类型上激活
    mask_rep   = out["arc_type"].isin(REPOSITION_TYPES)
    mask_tochg = out["arc_type"].eq("tochg")
    mask_occ   = out["arc_type"].eq("chg_occ")
    mask_step  = out["arc_type"].eq("chg_step")
    mask_sgate = out["arc_type"].eq("svc_gate")

    out.loc[~mask_rep,   "coef_rep"]         = 0.0
    out.loc[~mask_tochg, "coef_chg_travel"]  = 0.0
    out.loc[~mask_occ,   "coef_chg_occ"]     = 0.0
    # 奖励开关控制（若关闭则统一置 0）
    if not _get_flag("enable_service_reward", True):
        out["coef_svc_gate"] = 0.0
    else:
        out.loc[~mask_sgate, "coef_svc_gate"] = 0.0

    if not _get_flag("enable_reposition_reward", True):
        out["coef_rep_reward"] = 0.0
    else:
        out.loc[~mask_rep, "coef_rep_reward"] = 0.0

    if not _get_flag("enable_charging_reward", True):
        out["coef_chg_reward"] = 0.0
    else:
        out.loc[~(mask_occ | mask_step), "coef_chg_reward"] = 0.0

    # 汇总净成本（NaN→0）
    for c in ["coef_rep","coef_chg_travel","coef_chg_occ","coef_svc_gate","coef_rep_reward","coef_chg_reward"]:
        out[c] = _to_finite_float(out.get(c, 0.0), 0.0)

    # 为 idle 弧添加机会成本
    cfg = get_config()
    idle_opportunity_cost = float(cfg.costs_equity.idle_opportunity_cost)
    mask_idle = out["arc_type"].eq("idle")
    out.loc[mask_idle, "coef_idle"] = idle_opportunity_cost
    out.loc[~mask_idle, "coef_idle"] = 0.0

    out["coef_total"] = (
        out["coef_rep"] + out["coef_chg_travel"] + out["coef_chg_occ"] +
        out["coef_svc_gate"] + out["coef_rep_reward"] + out["coef_chg_reward"] + out["coef_idle"]
    )

    # 屏蔽“负成本自环”的 gate（防免费奖励）
    self_gate = (out["arc_type"].eq("svc_gate")) & (out["from_node_id"] == out["to_node_id"])
    if self_gate.any():
        if get_config().solver.verbose:
            print(f"[costs] zero-ing svc_gate self-loops: {int(self_gate.sum())}")
        out.loc[self_gate, ["coef_svc_gate", "coef_total"]] = 0.0

    return out


# ============================================================
# 5) 求解后评估（统一：基于“弧流量 + 系数”）
# ============================================================
def compute_operational_cost_from_arcflows(arc_flows: pd.DataFrame, arc_df_win: Optional[pd.DataFrame] = None) -> float:
    """
    给定“统一弧流量表”（至少含 arc_id, flow），返回运行成本：
      sum_a flow[a] * coef_total[a]
    若提供 arc_df_win，则动态附加 coef_total；否则需要 arc_flows 中已包含 coef_total。
    """
    if arc_flows is None or arc_flows.empty:
        return 0.0
    df = arc_flows.copy()

    if "coef_total" not in df.columns:
        if arc_df_win is None or arc_df_win.empty:
            raise ValueError("compute_operational_cost_from_arcflows: 需要 arc_df_win 来附加 coef_total。")
        # 将窗口弧与流量合并后附加成本
        arcs = attach_costs_to_arcs_for_window(arc_df_win)
        key = "arc_id" if "arc_id" in df.columns and "arc_id" in arcs.columns else None
        if not key:
            raise ValueError("需要 'arc_id' 以在 arc_flows 与 arc_df_win 之间对齐。")
        df = df.merge(arcs[["arc_id","coef_total"]], on="arc_id", how="left")

    df["flow"] = df["flow"].astype(float)
    df["coef_total"] = df["coef_total"].fillna(0.0).astype(float)
    return float((df["flow"] * df["coef_total"]).sum())


# ============================================================
# 6) 批量生成（静态模式所需的落盘系数）
# ============================================================
def build_all_coefficients() -> Dict[str, str]:
    ensure_dir(Path("data/intermediate"))

    rep_path = ""
    chg_tr_path = ""
    chg_occ_path = ""
    svc_gate_path = ""

    rep = build_reposition_cost_coefficients(save=True)
    if not rep.empty:
        rep_path = "data/intermediate/reposition_costs.parquet"

    chg_paths = build_charging_cost_coefficients(save=True)
    chg_tr_path = chg_paths.get("charging_travel_costs", "")
    chg_occ_path = chg_paths.get("charging_occupancy_costs", "")

    if _get_flag("enable_service_reward", True):
        svc = build_service_reward_coefficients(save=True)
        if not svc.empty:
            svc_gate_path = "data/intermediate/service_gate_rewards.parquet"

    # 可选：仍生成“未满足惩罚”以便分析（与 svc 奖励数值相同）
    un = build_unmet_penalty_coefficients(save=True)
    unmet_path = "data/intermediate/unmet_penalty.parquet" if not un.empty else ""

    return {
        "reposition_costs": rep_path,
        "charging_travel_costs": chg_tr_path,
        "charging_occupancy_costs": chg_occ_path,
        "service_gate_rewards": svc_gate_path,
        "unmet_penalty": unmet_path,
    }


if __name__ == "__main__":
    cfg = get_config()
    paths = build_all_coefficients()
    if cfg.solver.verbose:
        print("[costs] Built coefficients:")
        for k, v in paths.items():
            print(f"  - {k}: {v}")
