#!/usr/bin/env python3
"""
生成 base_i2k 数据脚本
根据 zone 和 station 的坐标生成 i,k,base_minutes,dist_km 数据

支持两种计算方式：
1. EUCLID: 欧几里得距离 + 平均速度
2. OSRM: 实际道路距离/时间（需要网络连接）

输出格式：
- CSV: base_i2k.csv
- Parquet: base_i2k.parquet

使用方法：
    python _04_create_base_i2k.py
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Tuple
import sys


# ===================== 配置常量 =====================
DEFAULT_METHOD = "EUCLID"  # 默认使用欧几里得距离

# 文件路径
ZONES_CSV = "../data/zones.csv"
STATIONS_CSV = "../data/stations.csv"
OUT_CSV = "../data/base_i2k.csv"
OUT_PARQUET = "../data/base_i2k.parquet"

# 欧几里得模式配置
SPEED_KMPH_EUCLID = 30.0  # km/h，用于把直线距离换算成分钟
ROW_BLOCK = 800  # zones 分块大小
COL_BLOCK = 800  # stations 分块大小

# OSRM 模式配置
OSRM_BASE_URL = "http://router.project-osrm.org"
OSRM_PROFILE = "driving"
OSRM_TABLE_BATCH = 80
OSRM_USE_TABLE = True
OSRM_CACHE_FILE = "data/cache/osm_distance_cache.pkl"
ORIG_BLOCK = 60  # OSRM: zones 起点分块大小
DEST_BLOCK = 60  # OSRM: stations 终点分块大小
FILL_MINUTES_IF_NAN = None  # 仅 OSRM 有效：若时长为 NaN，用该值（分钟）回填
# =====================================================


def _require_cols(df: pd.DataFrame, need: list[str]):
    """检查 DataFrame 是否包含必需的列"""
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"缺少列 {miss}；实际列：{list(df.columns)}")


def load_zones(zones_csv: str) -> pd.DataFrame:
    """加载 zones 数据"""
    z = pd.read_csv(zones_csv, dtype={"zone": str})
    _require_cols(z, ["zone", "lat", "lon"])
    z["lat"] = pd.to_numeric(z["lat"], errors="coerce")
    z["lon"] = pd.to_numeric(z["lon"], errors="coerce")
    z = z.dropna(subset=["lat", "lon"]).drop_duplicates(["zone"]).reset_index(drop=True)
    return z[["zone", "lat", "lon"]]


def load_stations(stations_csv: str) -> pd.DataFrame:
    """加载 stations 数据"""
    s = pd.read_csv(stations_csv, dtype={"k": str})
    # 兼容大小写/别名
    cols = {c.lower(): c for c in s.columns}
    kcol = cols.get("k") or cols.get("id") or cols.get("station_id")
    latc = cols.get("lat") or cols.get("latitude")
    lonc = cols.get("lon") or cols.get("lng") or cols.get("longitude")
    
    if not all([kcol, latc, lonc]):
        raise ValueError("stations CSV 需包含列 [k, lat, lon]（大小写或同义名均可）")
    
    s = s[[kcol, latc, lonc]].rename(columns={kcol: "k", latc: "lat", lonc: "lon"})
    s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    s = s.dropna(subset=["lat", "lon"]).drop_duplicates(["k"]).reset_index(drop=True)
    s["k"] = s["k"].astype(str)
    return s[["k", "lat", "lon"]]


def tile_ranges(n: int, r: int) -> Iterator[Tuple[int, int]]:
    """生成 [0,n) 的分块区间 [i0,i1)"""
    i = 0
    while i < n:
        j = min(n, i + r)
        yield i, j
        i = j


def build_by_euclid_stream(z: pd.DataFrame, st: pd.DataFrame, out_csv: str):
    """使用欧几里得距离计算并流式写入"""
    print(f"使用欧几里得距离计算，平均速度: {SPEED_KMPH_EUCLID} km/h")
    
    zid = z["zone"].to_numpy(str)     # (Nz,)
    zlat = z["lat"].to_numpy(float)
    zlon = z["lon"].to_numpy(float)

    kid = st["k"].to_numpy(str)       # (Nk,)
    klat = st["lat"].to_numpy(float)
    klon = st["lon"].to_numpy(float)

    Nz, Nk = len(z), len(st)
    print(f"Zones: {Nz}, Stations: {Nk}, 总组合数: {Nz * Nk:,}")

    R = 6371.0088  # 地球半径 km
    zlat_r = np.radians(zlat)
    zlon_r = np.radians(zlon)
    klat_r = np.radians(klat)
    klon_r = np.radians(klon)

    first_chunk = True
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    for i0, i1 in tile_ranges(Nz, ROW_BLOCK):
        # 行块（zones）
        lat_row = zlat_r[i0:i1][:, None]  # (R,1)
        lon_row = zlon_r[i0:i1][:, None]  # (R,1)
        ids_i = zid[i0:i1]              # (R,)

        for j0, j1 in tile_ranges(Nk, COL_BLOCK):
            # 列块（stations）
            lat_col = klat_r[None, j0:j1] # (1,C)
            lon_col = klon_r[None, j0:j1] # (1,C)
            ids_k = kid[j0:j1]          # (C,)

            # equirectangular 近似：直线距离
            dlat = lat_col - lat_row                  # (R,C)
            dlon = lon_col - lon_row                  # (R,C)
            x = dlon * np.cos((lat_row + lat_col)/2.) # (R,C)
            y = dlat
            dist_km = R * np.sqrt(x*x + y*y)          # (R,C)

            # 距离 -> 分钟
            base_minutes = (dist_km / max(1e-9, float(SPEED_KMPH_EUCLID))) * 60.0

            # 展平为列
            I = np.repeat(ids_i, j1-j0)              # (R*C,)
            K = np.tile(ids_k,  i1-i0)               # (R*C,)
            m = base_minutes.ravel()
            d = dist_km.ravel()

            df_chunk = pd.DataFrame({
                "i": I,
                "k": K,
                "base_minutes": m.astype(float),
                "dist_km": d.astype(float),
            })

            df_chunk.to_csv(out_csv, index=False, mode="w" if first_chunk else "a",
                            header=first_chunk)
            first_chunk = False

        print(f"[EUCLID] zones {i1}/{Nz} (~{i1/Nz:.1%}) 已写出...")

    print(f"[EUCLID] 完成：{out_csv}")


def build_by_osrm_stream(z: pd.DataFrame, st: pd.DataFrame, out_csv: str):
    """使用 OSRM 计算实际道路距离/时间并流式写入"""
    try:
        from osm_distance import OSRMDistanceCalculator, OSRMConfig
    except Exception as e:
        raise ImportError("无法导入 osm_distance.py，请将其放在同目录或 PYTHONPATH 中。") from e

    print(f"使用 OSRM 计算实际道路距离/时间")
    print(f"OSRM URL: {OSRM_BASE_URL}")

    cfg = OSRMConfig(
        base_url=OSRM_BASE_URL,
        profile=OSRM_PROFILE,
        table_batch=OSRM_TABLE_BATCH,
        use_table_api=OSRM_USE_TABLE,
    )
    calc = OSRMDistanceCalculator(osrm_cfg=cfg, cache_file=OSRM_CACHE_FILE)

    O = z.rename(columns={"zone": "id"})[["id", "lat", "lon"]].copy()
    D = st.rename(columns={"k": "id"})[["id", "lat", "lon"]].copy()
    Nz, Nk = len(O), len(D)
    print(f"Zones: {Nz}, Stations: {Nk}, 总组合数: {Nz * Nk:,}")

    first_chunk = True
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    for i0, i1 in tile_ranges(Nz, ORIG_BLOCK):
        O_blk = O.iloc[i0:i1].copy()
        for j0, j1 in tile_ranges(Nk, DEST_BLOCK):
            D_blk = D.iloc[j0:j1].copy()

            df = calc.matrix(
                origins=O_blk, destinations=D_blk,
                origin_id_col="id", dest_id_col="id",
                lat_col="lat", lon_col="lon"
            )
            # df: origin_id, destination_id, distance_km, duration_min, source
            df = df.rename(columns={
                "origin_id": "i", "destination_id": "k",
                "duration_min": "base_minutes", "distance_km": "dist_km"
            })[["i", "k", "base_minutes", "dist_km"]]

            if FILL_MINUTES_IF_NAN is not None:
                df["base_minutes"] = pd.to_numeric(df["base_minutes"], errors="coerce").fillna(float(FILL_MINUTES_IF_NAN))

            df.to_csv(out_csv, index=False, mode="w" if first_chunk else "a",
                      header=first_chunk)
            first_chunk = False

        print(f"[OSRM] zones {i1}/{Nz} (~{i1/Nz:.1%}) 已写出...")

    print(f"[OSRM] 完成：{out_csv}")


def main():
    # 硬编码配置
    method = DEFAULT_METHOD
    zones_file = ZONES_CSV
    stations_file = STATIONS_CSV
    output_csv = OUT_CSV
    output_parquet = OUT_PARQUET

    print(f"开始生成 base_i2k 数据...")
    print(f"方法: {method}")
    print(f"Zones 文件: {zones_file}")
    print(f"Stations 文件: {stations_file}")
    print(f"输出 CSV: {output_csv}")
    print(f"输出 Parquet: {output_parquet}")

    try:
        # 加载数据
        print("\n加载数据...")
        zones = load_zones(zones_file)
        stations = load_stations(stations_file)
        print(f"成功加载 {len(zones)} 个 zones 和 {len(stations)} 个 stations")

        # 计算距离
        print(f"\n开始计算...")
        if method.upper() == "EUCLID":
            build_by_euclid_stream(zones, stations, output_csv)
        elif method.upper() == "OSRM":
            build_by_osrm_stream(zones, stations, output_csv)
        else:
            raise ValueError("METHOD 必须为 'EUCLID' 或 'OSRM'")

        # 转换为 Parquet 格式
        print(f"\n转换为 Parquet 格式...")
        df = pd.read_csv(output_csv)
        df.to_parquet(output_parquet, index=False)
        print(f"Parquet 文件已保存到: {output_parquet}")

        # 简单检查：读取前几行
        print(f"\n检查输出文件...")
        head = df.head(5)
        print("前5行数据:")
        print(head)
        
        # 统计信息
        total_rows = len(df)
        print(f"\n总计生成 {total_rows:,} 行数据")
        print(f"CSV 文件已保存到: {output_csv}")
        print(f"Parquet 文件已保存到: {output_parquet}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
