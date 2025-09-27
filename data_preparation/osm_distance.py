# osm_distance.py
"""
OSRM 距离与时间估算（可与 SAEV 优化流水线协作）
- 直接生成 base_ij.parquet / base_i2k.parquet，供 CostProvider/04_arc_generators 使用
- 支持 OSRM /table 批处理，失败时回退 /route，再回退 Haversine
- 与 config.py 集成，自动读取 OD/zones/stations
"""

from __future__ import annotations
import os
import time
import math
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from config import get_config
except Exception:
    get_config = None  # 允许独立使用

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------
# 工具
# ---------------------------

def _first_present(cols: Iterable[str], cands: List[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in lowers:
            return lowers[cand.lower()]
    return None

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面距离（km）"""
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------
# OSRM 客户端
# ---------------------------

@dataclass
class OSRMConfig:
    base_url: str = "http://router.project-osrm.org"
    profile: str = "driving"
    request_delay: float = 0.05
    max_retries: int = 3
    timeout_sec: int = 15
    user_agent: str = "saev-osrm-client/1.0"
    use_table_api: bool = True
    table_batch: int = 80  # 每批最多坐标数量（orig+dest），注意公用服务的限制
    table_annotations: str = "duration,distance"  # OSRM table 支持 duration,distance


class OSRMDistanceCalculator:
    """
    OpenStreetMap distance calculator using OSRM API
    - 支持 /table 批处理 & /route 单对请求
    - 透明缓存（pickle），并支持把结果另存为 parquet 小表
    """

    def __init__(self,
                 osrm_cfg: OSRMConfig = OSRMConfig(),
                 cache_file: str = "data/cache/osm_distance_cache.pkl"):
        self.cfg = osrm_cfg
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_file
        self.cache = self._load_cache()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.user_agent})

    # -------- 缓存 --------
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _key(self, o_id: str, d_id: str, profile: str) -> str:
        return f"{profile}::{o_id}::{d_id}"

    def get_cache_stats(self) -> Dict:
        return {
            "total_entries": len(self.cache),
            "cache_file_size_mb": os.path.getsize(self.cache_file) / (1024*1024) if os.path.exists(self.cache_file) else 0,
        }

    def clear_cache(self):
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("OSRM distance cache cleared")

    # -------- 基础请求 --------
    def _table(self, coords: List[Tuple[float, float]],
               sources: List[int], destinations: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        调用 OSRM /table 接口，返回 (durations_matrix_minutes, distances_matrix_km)
        coords: [(lon,lat), ...]  注意 OSRM 顺序 = lon,lat
        sources/destinations: index 列表
        """
        # 构造 url
        coord_str = ";".join([f"{c[0]:.6f},{c[1]:.6f}" for c in coords])
        url = f"{self.cfg.base_url}/table/v1/{self.cfg.profile}/{coord_str}"
        params = {
            "sources": ";".join(str(i) for i in sources),
            "destinations": ";".join(str(i) for i in destinations),
            "annotations": self.cfg.table_annotations,
        }

        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == "Ok":
                        durs = np.array(data.get("durations", []), dtype=float)  # 秒
                        dists = np.array(data.get("distances", []), dtype=float)  # 米
                        if durs.size == 0:
                            raise ValueError("Empty durations from OSRM table")
                        return durs/60.0, dists/1000.0
                    else:
                        logger.warning(f"OSRM table error: {data.get('message')}")
                else:
                    logger.warning(f"OSRM table HTTP {resp.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"OSRM table failed (attempt {attempt+1}): {e}")
            time.sleep(self.cfg.request_delay * (attempt + 1))
        raise RuntimeError("OSRM table failed after retries")

    def _route(self, o: Tuple[float, float], d: Tuple[float, float]) -> Tuple[Optional[float], Optional[float]]:
        """
        单对 /route 请求（返回 distance_km, duration_min）
        参数为 (lat,lon)
        """
        lat1, lon1 = o
        lat2, lon2 = d
        url = f"{self.cfg.base_url}/route/v1/{self.cfg.profile}/{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
        params = {"overview": "false", "alternatives": "false", "steps": "false"}
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == "Ok" and data.get("routes"):
                        r = data["routes"][0]
                        return r["distance"]/1000.0, r["duration"]/60.0
                    else:
                        logger.debug(f"OSRM route not ok: {data.get('message')}")
                else:
                    logger.debug(f"OSRM route HTTP {resp.status_code}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"OSRM route failed (attempt {attempt+1}): {e}")
            time.sleep(self.cfg.request_delay * (attempt + 1))
        return None, None

    # -------- 高层接口：矩阵计算（含缓存与回退） --------
    def matrix(self,
               origins: pd.DataFrame, destinations: pd.DataFrame,
               origin_id_col: Optional[str] = None, dest_id_col: Optional[str] = None,
               lat_col: Optional[str] = None, lon_col: Optional[str] = None) -> pd.DataFrame:
        """
        计算 origins×destinations 的距离/时间（优先 OSRM table）
        返回列: origin_id, destination_id, distance_km, duration_min, source
        """
        # 列名推断
        origin_id_col = origin_id_col or _first_present(origins.columns, ["zone", "id", "zone_id", "origin_id"])
        dest_id_col = dest_id_col or _first_present(destinations.columns, ["zone", "id", "zone_id", "destination_id"])
        lat_col = lat_col or _first_present(origins.columns, ["lat", "latitude"])
        lon_col = lon_col or _first_present(origins.columns, ["lon", "lng", "longitude"])
        lat2_col = _first_present(destinations.columns, ["lat", "latitude"])
        lon2_col = _first_present(destinations.columns, ["lon", "lng", "longitude"])
        if None in (origin_id_col, dest_id_col, lat_col, lon_col, lat2_col, lon2_col):
            raise ValueError("origins/destinations 需要列 [id, lat, lon]（可同义）")

        # 组装坐标
        o_df = origins[[origin_id_col, lat_col, lon_col]].copy()
        d_df = destinations[[dest_id_col, lat2_col, lon2_col]].copy()
        o_df.columns = ["oid", "olat", "olon"]
        d_df.columns = ["did", "dlat", "dlon"]

        # 先用缓存命中
        results: List[Dict] = []
        to_query_pairs: List[Tuple[str, str]] = []
        for _, ro in o_df.iterrows():
            for _, rd in d_df.iterrows():
                key = self._key(str(ro["oid"]), str(rd["did"]), self.cfg.profile)
                if key in self.cache:
                    dist_km, dur_min = self.cache[key]
                    results.append({"origin_id": ro["oid"], "destination_id": rd["did"],
                                    "distance_km": dist_km, "duration_min": dur_min, "source": "cache"})
                else:
                    to_query_pairs.append((ro["oid"], rd["did"]))

        # 如果全部命中直接返回
        if not to_query_pairs:
            return pd.DataFrame(results)

        # 需要查询的坐标索引
        # 构造 indices，尽可能使用 /table 批处理
        if self.cfg.use_table_api:
            try:
                # 为 /table 构造去重坐标集合，建立映射
                o_idx = {oid: idx for idx, oid in enumerate(o_df["oid"].tolist())}
                d_idx = {did: idx for idx, did in enumerate(d_df["did"].tolist())}
                coords = []
                # 顺序：先 origins 再 destinations
                for _, r in o_df.iterrows():
                    coords.append((float(r["olon"]), float(r["olat"])))
                base_dest_offset = len(coords)
                for _, r in d_df.iterrows():
                    coords.append((float(r["dlon"]), float(r["dlat"])))

                # 分批（origins 批 × destinations 批），避免坐标长度过大
                o_list = list(o_df["oid"])
                d_list = list(d_df["did"])
                batch = max(10, int(self.cfg.table_batch // 2))  # 粗略切半
                for i0 in range(0, len(o_list), batch):
                    o_chunk = o_list[i0:i0+batch]
                    s_idx = [o_idx[oid] for oid in o_chunk]
                    for j0 in range(0, len(d_list), batch):
                        d_chunk = d_list[j0:j0+batch]
                        t_idx = [base_dest_offset + d_idx[did] for did in d_chunk]
                        # 请求
                        durs_min, dists_km = self._table(coords, s_idx, t_idx)
                        # 写入结果与缓存
                        for ii, oid in enumerate(o_chunk):
                            for jj, did in enumerate(d_chunk):
                                dur = float(durs_min[ii, jj]) if np.isfinite(durs_min[ii, jj]) else np.nan
                                dist = float(dists_km[ii, jj]) if np.isfinite(dists_km[ii, jj]) else np.nan
                                if not np.isfinite(dur) or not np.isfinite(dist):
                                    # 无路，后续用 /route 或 Haversine 填
                                    continue
                                key = self._key(str(oid), str(did), self.cfg.profile)
                                self.cache[key] = (dist, dur)
                                results.append({"origin_id": oid, "destination_id": did,
                                                "distance_km": dist, "duration_min": dur, "source": "osrm_table"})
                        time.sleep(self.cfg.request_delay)
                self._save_cache()
            except Exception as e:
                logger.warning(f"/table failed, fallback to per-pair route: {e}")

        # 对仍未得到的数据逐对 /route，再回退 Haversine
        have = {(r["origin_id"], r["destination_id"]) for r in results}
        for _, ro in o_df.iterrows():
            for _, rd in d_df.iterrows():
                key = (ro["oid"], rd["did"])
                if key in have:
                    continue
                cache_key = self._key(str(ro["oid"]), str(rd["did"]), self.cfg.profile)
                if cache_key in self.cache:
                    dist, dur = self.cache[cache_key]
                    src = "cache"
                else:
                    dist, dur = self._route((ro["olat"], ro["olon"]), (rd["dlat"], rd["dlon"]))
                    src = "osrm_route"
                    if dist is None or dur is None:
                        # Haversine 回退
                        dist = haversine_km(ro["olat"], ro["olon"], rd["dlat"], rd["dlon"])
                        # 无法给出稳定车速，这里仅保存距离；时长置 NaN（由上游转换/回归）
                        dur = np.nan
                        src = "haversine"
                    else:
                        self.cache[cache_key] = (dist, dur)
                        self._save_cache()
                    time.sleep(self.cfg.request_delay)
                results.append({"origin_id": ro["oid"], "destination_id": rd["did"],
                                "distance_km": float(dist), "duration_min": float(dur) if np.isfinite(dur) else np.nan,
                                "source": src})
        return pd.DataFrame(results)


# ---------------------------
# 与流水线协作的产出：base_ij/base_i2k
# ---------------------------

def build_base_ij_from_od(
    calc: OSRMDistanceCalculator,
    od_path: Optional[str] = None,
    zones_path: Optional[str] = None,
    out_path: str = "data/base_ij.parquet",
    drop_self_loop: bool = False,
    fill_minutes_if_nan: Optional[float] = None
) -> pd.DataFrame:
    """
    用 OD 中出现的 (i,j) 与 zones 坐标，批量估算 base_minutes/dist_km，写出 base_ij.parquet
    - base_minutes = duration_min（若 NaN 且 fill_minutes_if_nan 非空，则用该值）
    """
    # 读 config
    if get_config is not None:
        cfg = get_config()
        od_path = od_path or cfg.paths.od_matrix
        zones_path = zones_path or cfg.paths.zones if hasattr(cfg.paths, "zones") else None

    if od_path is None or zones_path is None:
        raise ValueError("需要提供 od_path 与 zones_path 或在 config.paths 中配置")

    od = pd.read_parquet(od_path, columns=["t", "i", "j"])  # 只需要唯一键
    kij = od[["i", "j"]].drop_duplicates().reset_index(drop=True)
    if drop_self_loop:
        kij = kij[kij["i"] != kij["j"]].reset_index(drop=True)

    zones = pd.read_csv(zones_path)
    col_id = _first_present(zones.columns, ["zone", "id", "zone_id"])
    col_lat = _first_present(zones.columns, ["lat", "latitude"])
    col_lon = _first_present(zones.columns, ["lon", "lng", "longitude"])
    if None in (col_id, col_lat, col_lon):
        raise ValueError("zones.csv 需要列 [zone/id, lat, lon]")

    z = zones[[col_id, col_lat, col_lon]].drop_duplicates()
    z.columns = ["zone", "lat", "lon"]

    # 准备 origins/destinations
    orig = kij.merge(z, left_on="i", right_on="zone", how="left").drop(columns=["zone"])
    dest = kij.merge(z, left_on="j", right_on="zone", how="left").drop(columns=["zone"])
    # 去重
    O = orig[["i", "lat", "lon"]].drop_duplicates().rename(columns={"i": "zone"})
    D = dest[["j", "lat", "lon"]].drop_duplicates().rename(columns={"j": "zone"})

    df = calc.matrix(
        origins=O.rename(columns={"zone": "id"}),
        destinations=D.rename(columns={"zone": "id"}),
        origin_id_col="id", dest_id_col="id", lat_col="lat", lon_col="lon"
    )
    # df: origin_id, destination_id, distance_km, duration_min, source
    out = df.rename(columns={"origin_id": "i", "destination_id": "j"})[["i", "j", "distance_km", "duration_min"]]
    out = out.merge(kij, on=["i", "j"], how="right")  # 确保每个 (i,j) 都有行
    # base_minutes
    if fill_minutes_if_nan is not None:
        out["duration_min"] = out["duration_min"].fillna(float(fill_minutes_if_nan))
    out = out.rename(columns={"duration_min": "base_minutes", "distance_km": "dist_km"})
    out = out[["i", "j", "base_minutes", "dist_km"]].drop_duplicates().reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(f"[base_ij] rows={len(out)} saved -> {out_path}")
    return out


def build_base_i2k_from_zones_stations(
    calc: OSRMDistanceCalculator,
    zones_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    out_path: str = "data/base_i2k.parquet",
    drop_missing: bool = True,
    fill_minutes_if_nan: Optional[float] = None
) -> pd.DataFrame:
    """
    区→站的基准时长/距离（用于 CostProvider.batch_to_station）
    输出列: [i, k, base_minutes, dist_km]
    """
    if get_config is not None:
        cfg = get_config()
        zones_path = zones_path or cfg.paths.zones if hasattr(cfg.paths, "zones") else None
        stations_path = stations_path or cfg.paths.stations

    if zones_path is None or stations_path is None:
        raise ValueError("需要提供 zones_path 与 stations_path 或在 config.paths 中配置")

    zones = pd.read_csv(zones_path)
    col_id = _first_present(zones.columns, ["zone", "id", "zone_id"])
    col_lat = _first_present(zones.columns, ["lat", "latitude"])
    col_lon = _first_present(zones.columns, ["lon", "lng", "longitude"])
    if None in (col_id, col_lat, col_lon):
        raise ValueError("zones.csv 需要列 [zone/id, lat, lon]")
    z = zones[[col_id, col_lat, col_lon]].drop_duplicates()
    z.columns = ["i", "lat", "lon"]

    st = pd.read_csv(stations_path)
    col_k = _first_present(st.columns, ["k", "station", "station_id"])
    col_slat = _first_present(st.columns, ["lat", "latitude"])
    col_slon = _first_present(st.columns, ["lon", "lng", "longitude"])
    if None in (col_k, col_slat, col_slon):
        raise ValueError("stations.csv 需要列 [k, lat, lon]")
    s = st[[col_k, col_slat, col_slon]].drop_duplicates()
    s.columns = ["k", "lat", "lon"]

    df = calc.matrix(
        origins=z.rename(columns={"i": "id"}),
        destinations=s.rename(columns={"k": "id"}),
        origin_id_col="id", dest_id_col="id", lat_col="lat", lon_col="lon"
    )
    out = df.rename(columns={"origin_id": "i", "destination_id": "k"})[["i", "k", "distance_km", "duration_min"]]
    if fill_minutes_if_nan is not None:
        out["duration_min"] = out["duration_min"].fillna(float(fill_minutes_if_nan))
    out = out.rename(columns={"duration_min": "base_minutes", "distance_km": "dist_km"})
    out = out[["i", "k", "base_minutes", "dist_km"]].drop_duplicates().reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(f"[base_i2k] rows={len(out)} saved -> {out_path}")
    return out


# ---------------------------
# CLI
# ---------------------------

def _default_calc_from_config() -> OSRMDistanceCalculator:
    # 允许从 config 读取自定义 OSRM base_url/profile
    osrm_cfg = OSRMConfig()
    if get_config is not None:
        cfg = get_config()
        # 非必须：如果 cfg.osrm.* 存在就覆盖
        if hasattr(cfg, "osrm"):
            osrm_cfg.base_url = cfg.osrm.base_url if hasattr(cfg.osrm, "base_url") else osrm_cfg.base_url
            osrm_cfg.profile = cfg.osrm.profile if hasattr(cfg.osrm, "profile") else osrm_cfg.profile
            osrm_cfg.table_batch = cfg.osrm.table_batch if hasattr(cfg.osrm, "table_batch") else osrm_cfg.table_batch
            osrm_cfg.use_table_api = cfg.osrm.use_table_api if hasattr(cfg.osrm, "use_table_api") else osrm_cfg.use_table_api
    return OSRMDistanceCalculator(osrm_cfg=osrm_cfg, cache_file="data/cache/osm_distance_cache.pkl")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build base_ij/base_i2k from OSRM for SAEV pipeline.")
    parser.add_argument("--task", choices=["base_ij", "base_i2k"], required=True)
    parser.add_argument("--od", help="Path to od_matrix.parquet (optional; default from config)")
    parser.add_argument("--zones", help="Path to zones.csv (optional; default from config)")
    parser.add_argument("--stations", help="Path to stations.csv (optional; default from config)")
    parser.add_argument("--out", help="Output parquet path (optional; default data/base_ij.parquet or data/base_i2k.parquet)")
    parser.add_argument("--drop-self-loop", action="store_true", help="Drop i==j (for base_ij)")
    parser.add_argument("--fill-minutes-if-nan", type=float, default=None, help="If OSRM duration is NaN, use this value (minutes).")
    args = parser.parse_args()

    calc = _default_calc_from_config()

    if args.task == "base_ij":
        out = args.out or "data/base_ij.parquet"
        build_base_ij_from_od(
            calc,
            od_path=args.od, zones_path=args.zones, out_path=out,
            drop_self_loop=args.drop_self_loop,
            fill_minutes_if_nan=args.fill_minutes_if_nan
        )
    else:
        out = args.out or "data/base_i2k.parquet"
        build_base_i2k_from_zones_stations(
            calc,
            zones_path=args.zones, stations_path=args.stations, out_path=out,
            fill_minutes_if_nan=args.fill_minutes_if_nan
        )
