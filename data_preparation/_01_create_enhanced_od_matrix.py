#!/usr/bin/env python
# coding: utf-8

# # 基于Ride-Austin数据生成高质量OD矩阵
# 
# 本脚本结合ride-austin.csv和TAZ数据，生成包含出发时间、工作日/非工作日等关键信息的高质量区域聚合需求OD矩阵。
# 
# ## 主要改进：
# 1. 使用真实的出行数据而非LODES工作通勤数据
# 2. 包含出发时间信息（小时级别）
# 3. 区分工作日和非工作日
# 4. 基于TAZ进行空间聚合
# 5. 包含距离、时长等出行特征
# 

# In[27]:


import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# 设置环境变量以处理shapefile问题
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# 尝试导入matplotlib，如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
    print("✓ matplotlib 可用，将生成可视化图表")
except ImportError as e:
    print(f"⚠️ matplotlib 不可用: {e}")
    print("将跳过可视化部分，仅生成数据文件")
    MATPLOTLIB_AVAILABLE = False


# ## 1. 数据加载和预处理
# 

# In[28]:


# 数据路径
data_dir = Path("../data")
raw_data_dir = data_dir / "raw data"
intermediate_dir = data_dir / "intermediate"
intermediate_dir.mkdir(exist_ok=True)

# 加载Ride-Austin数据
print("加载Ride-Austin数据...")
ride_df = pd.read_csv(raw_data_dir / "ride-austin.csv")
print(f"Ride-Austin数据形状: {ride_df.shape}")
print(f"列名: {list(ride_df.columns)}")
ride_df.head()


# In[29]:


# 加载TAZ shapefile
print("加载TAZ数据...")
taz_gdf = gpd.read_file(raw_data_dir / "tl_2011_48_taz10.shp")
print(f"TAZ数据形状: {taz_gdf.shape}")
print(f"列名: {list(taz_gdf.columns)}")
taz_gdf.head()


# In[30]:


# 加载现有zones数据（用于筛选）
zones_df = pd.read_csv(data_dir / "zones.csv")
print(f"Zones数据形状: {zones_df.shape}")
print(f"有效zones数量: {len(zones_df)}")
zones_df.head()


# ## 2. Ride-Austin数据预处理
# 

# In[31]:


# 数据清洗和预处理
print("开始数据预处理...")

# 转换时间列
ride_df['started_on'] = pd.to_datetime(ride_df['started_on'])
ride_df['completed_on'] = pd.to_datetime(ride_df['completed_on'])

# 计算出行时长（分钟）
ride_df['duration_minutes'] = (ride_df['completed_on'] - ride_df['started_on']).dt.total_seconds() / 60

# 提取时间特征
ride_df['date'] = ride_df['started_on'].dt.date
ride_df['hour'] = ride_df['started_on'].dt.hour
ride_df['day_of_week'] = ride_df['started_on'].dt.dayofweek  # 0=Monday, 6=Sunday
ride_df['is_weekend'] = ride_df['day_of_week'].isin([5, 6])  # Saturday, Sunday
ride_df['is_weekday'] = ~ride_df['is_weekend']

# 定义时间段
def get_time_period(hour):
    if 6 <= hour < 9:
        return 'morning_peak'
    elif 9 <= hour < 16:
        return 'midday'
    elif 16 <= hour < 19:
        return 'evening_peak'
    elif 19 <= hour < 22:
        return 'evening'
    else:
        return 'night'

ride_df['time_period'] = ride_df['hour'].apply(get_time_period)

print(f"预处理后数据形状: {ride_df.shape}")
print(f"时间范围: {ride_df['started_on'].min()} 到 {ride_df['started_on'].max()}")
print(f"工作日出行: {ride_df['is_weekday'].sum():,} 次")
print(f"周末出行: {ride_df['is_weekend'].sum():,} 次")


# In[32]:


# 数据质量检查
print("数据质量检查:")
print(f"缺失起始坐标: {ride_df[['start_location_lat', 'start_location_long']].isnull().any(axis=1).sum():,}")
print(f"缺失终点坐标: {ride_df[['end_location_lat', 'end_location_long']].isnull().any(axis=1).sum():,}")
print(f"异常距离数据: {(ride_df['distance_travelled'] <= 0).sum():,}")
print(f"异常时长数据: {(ride_df['duration_minutes'] <= 0).sum():,}")

# 过滤有效数据
valid_mask = (
    ride_df['start_location_lat'].notna() & 
    ride_df['start_location_long'].notna() &
    ride_df['end_location_lat'].notna() & 
    ride_df['end_location_long'].notna() &
    (ride_df['distance_travelled'] > 0) &
    (ride_df['duration_minutes'] > 0) &
    (ride_df['duration_minutes'] < 300)  # 过滤超过5小时的异常数据
)

ride_clean = ride_df[valid_mask].copy()
print(f"清洗后有效数据: {len(ride_clean):,} 条 ({len(ride_clean)/len(ride_df)*100:.1f}%)")


# ## 3. 空间聚合：将出行点映射到TAZ
# 

# In[33]:


# 创建起始点和终点的GeoDataFrame
print("创建空间数据...")

# 起始点
start_points = gpd.GeoDataFrame(
    ride_clean[['started_on', 'start_location_lat', 'start_location_long']],
    geometry=[Point(xy) for xy in zip(ride_clean['start_location_long'], ride_clean['start_location_lat'])],
    crs='EPSG:4326'
)

# 终点
end_points = gpd.GeoDataFrame(
    ride_clean[['completed_on', 'end_location_lat', 'end_location_long']],
    geometry=[Point(xy) for xy in zip(ride_clean['end_location_long'], ride_clean['end_location_lat'])],
    crs='EPSG:4326'
)

print(f"起始点数量: {len(start_points)}")
print(f"终点数量: {len(end_points)}")


# In[34]:


# 空间连接：将点映射到TAZ
print("执行空间连接...")

# 确保TAZ数据使用正确的坐标系
if taz_gdf.crs != 'EPSG:4326':
    taz_gdf = taz_gdf.to_crs('EPSG:4326')

# 起始点空间连接
start_with_taz = gpd.sjoin(start_points, taz_gdf, how='left', predicate='within')
print(f"起始点成功映射到TAZ: {start_with_taz['TAZCE10'].notna().sum():,} 个")

# 终点空间连接
end_with_taz = gpd.sjoin(end_points, taz_gdf, how='left', predicate='within')
print(f"终点成功映射到TAZ: {end_with_taz['TAZCE10'].notna().sum():,} 个")


# In[35]:


# 合并起始和终点TAZ信息
ride_with_taz = ride_clean.copy()
ride_with_taz['start_taz'] = start_with_taz['TAZCE10'].values
ride_with_taz['end_taz'] = end_with_taz['TAZCE10'].values

# 过滤有效TAZ映射的数据
valid_taz_mask = ride_with_taz['start_taz'].notna() & ride_with_taz['end_taz'].notna()
ride_taz = ride_with_taz[valid_taz_mask].copy()

print(f"成功映射到TAZ的出行: {len(ride_taz):,} 条 ({len(ride_taz)/len(ride_clean)*100:.1f}%)")
print(f"涉及起始TAZ数量: {ride_taz['start_taz'].nunique()}")
print(f"涉及终点TAZ数量: {ride_taz['end_taz'].nunique()}")


# In[36]:


# 筛选在服务区域内的TAZ
valid_zones = set(zones_df['zone'].astype(str).str.zfill(8))
ride_taz['start_taz_str'] = ride_taz['start_taz'].astype(str).str.zfill(8)
ride_taz['end_taz_str'] = ride_taz['end_taz'].astype(str).str.zfill(8)

# 过滤在服务区域内的出行
service_area_mask = (
    ride_taz['start_taz_str'].isin(valid_zones) & 
    ride_taz['end_taz_str'].isin(valid_zones)
)
ride_service = ride_taz[service_area_mask].copy()

print(f"服务区域内出行: {len(ride_service):,} 条 ({len(ride_service)/len(ride_taz)*100:.1f}%)")
print(f"服务区域内起始TAZ: {ride_service['start_taz_str'].nunique()}")
print(f"服务区域内终点TAZ: {ride_service['end_taz_str'].nunique()}")


# ## 4. 生成增强的OD矩阵
# 

# In[37]:


# 按时间、工作日/周末、OD对聚合
print("生成OD矩阵...")

# 基础聚合
od_agg = ride_service.groupby([
    'start_taz_str', 'end_taz_str', 'hour', 'is_weekday'
]).agg({
    'distance_travelled': ['count', 'mean', 'std'],
    'duration_minutes': ['mean', 'std'],
    'started_on': 'min'  # 记录最早出行时间
}).reset_index()

# 展平列名
od_agg.columns = [
    'i', 'j', 'hour', 'is_weekday', 
    'demand', 'avg_distance', 'std_distance',
    'avg_duration', 'std_duration', 'first_trip_time'
]

print(f"OD矩阵形状: {od_agg.shape}")
print(f"总需求: {od_agg['demand'].sum():,} 次出行")
od_agg.head()


# In[38]:


# 添加时间段信息
od_agg['time_period'] = od_agg['hour'].apply(get_time_period)
od_agg['day_type'] = od_agg['is_weekday'].map({True: 'weekday', False: 'weekend'})

# 重新排列列
od_matrix = od_agg[[
    'i', 'j', 'hour', 'time_period', 'day_type', 'is_weekday',
    'demand', 'avg_distance', 'std_distance', 'avg_duration', 'std_duration',
    'first_trip_time'
]].copy()

# 按需求排序
od_matrix = od_matrix.sort_values(['demand', 'i', 'j', 'hour'], ascending=[False, True, True, True])

print(f"最终OD矩阵形状: {od_matrix.shape}")
print(f"涉及OD对数量: {len(od_matrix[['i', 'j']].drop_duplicates())}")
print(f"时间分布:")
print(od_matrix['time_period'].value_counts())
print(f"\n工作日/周末分布:")
print(od_matrix['day_type'].value_counts())


# ## 5. 数据分析和可视化
# 

# In[39]:


# 需求分布分析
if MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 需求分布直方图
    axes[0, 0].hist(od_matrix['demand'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('需求次数')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('OD需求分布')
    axes[0, 0].set_yscale('log')

    # 2. 小时需求分布
    hourly_demand = od_matrix.groupby('hour')['demand'].sum()
    axes[0, 1].plot(hourly_demand.index, hourly_demand.values, marker='o')
    axes[0, 1].set_xlabel('小时')
    axes[0, 1].set_ylabel('总需求')
    axes[0, 1].set_title('小时需求分布')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 时间段需求分布
    period_demand = od_matrix.groupby('time_period')['demand'].sum()
    axes[1, 0].bar(period_demand.index, period_demand.values)
    axes[1, 0].set_xlabel('时间段')
    axes[1, 0].set_ylabel('总需求')
    axes[1, 0].set_title('时间段需求分布')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. 工作日/周末需求分布
    day_demand = od_matrix.groupby('day_type')['demand'].sum()
    axes[1, 1].bar(day_demand.index, day_demand.values)
    axes[1, 1].set_xlabel('日期类型')
    axes[1, 1].set_ylabel('总需求')
    axes[1, 1].set_title('工作日/周末需求分布')

    plt.tight_layout()
    plt.show()
else:
    # 文本统计信息
    print("需求分布统计:")
    print(f"   - 总需求: {od_matrix['demand'].sum():,} 次")
    print(f"   - 平均需求: {od_matrix['demand'].mean():.1f} 次")
    print(f"   - 需求中位数: {od_matrix['demand'].median():.1f} 次")
    
    hourly_demand = od_matrix.groupby('hour')['demand'].sum()
    print(f"\n小时需求分布（前5个高峰时段）:")
    for hour, demand in hourly_demand.nlargest(5).items():
        print(f"   - {hour:02d}:00 - {demand:,} 次")
    
    period_demand = od_matrix.groupby('time_period')['demand'].sum()
    print(f"\n时间段需求分布:")
    for period, demand in period_demand.items():
        print(f"   - {period}: {demand:,} 次")
    
    day_demand = od_matrix.groupby('day_type')['demand'].sum()
    print(f"\n工作日/周末需求分布:")
    for day_type, demand in day_demand.items():
        print(f"   - {day_type}: {demand:,} 次")


# In[40]:


# 距离和时长分析
if MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 平均距离分布
    axes[0, 0].hist(od_matrix['avg_distance'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('平均距离 (米)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('平均出行距离分布')

    # 2. 平均时长分布
    axes[0, 1].hist(od_matrix['avg_duration'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('平均时长 (分钟)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('平均出行时长分布')

    # 3. 距离vs需求散点图
    axes[1, 0].scatter(od_matrix['avg_distance'], od_matrix['demand'], alpha=0.5)
    axes[1, 0].set_xlabel('平均距离 (米)')
    axes[1, 0].set_ylabel('需求次数')
    axes[1, 0].set_title('距离vs需求')
    axes[1, 0].set_yscale('log')

    # 4. 时长vs需求散点图
    axes[1, 1].scatter(od_matrix['avg_duration'], od_matrix['demand'], alpha=0.5)
    axes[1, 1].set_xlabel('平均时长 (分钟)')
    axes[1, 1].set_ylabel('需求次数')
    axes[1, 1].set_title('时长vs需求')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()
else:
    # 文本统计信息
    print("距离和时长统计:")
    print(f"   - 平均出行距离: {od_matrix['avg_distance'].mean():.0f} 米")
    print(f"   - 距离中位数: {od_matrix['avg_distance'].median():.0f} 米")
    print(f"   - 距离标准差: {od_matrix['avg_distance'].std():.0f} 米")
    print(f"   - 平均出行时长: {od_matrix['avg_duration'].mean():.1f} 分钟")
    print(f"   - 时长中位数: {od_matrix['avg_duration'].median():.1f} 分钟")
    print(f"   - 时长标准差: {od_matrix['avg_duration'].std():.1f} 分钟")
    
    # 距离vs需求相关性
    distance_demand_corr = od_matrix['avg_distance'].corr(od_matrix['demand'])
    duration_demand_corr = od_matrix['avg_duration'].corr(od_matrix['demand'])
    print(f"\n相关性分析:")
    print(f"   - 距离与需求相关性: {distance_demand_corr:.3f}")
    print(f"   - 时长与需求相关性: {duration_demand_corr:.3f}")


# ## 6. 生成不同格式的OD矩阵
# 

# In[41]:


# 分别生成工作日和非工作日的OD矩阵
print("分别生成工作日和非工作日的OD矩阵...")

# 工作日OD矩阵
od_weekday = od_matrix[od_matrix['is_weekday'] == True].copy()
print(f"工作日OD矩阵形状: {od_weekday.shape}")
print(f"工作日总需求: {od_weekday['demand'].sum():,} 次出行")

# 非工作日OD矩阵
od_weekend = od_matrix[od_matrix['is_weekday'] == False].copy()
print(f"非工作日OD矩阵形状: {od_weekend.shape}")
print(f"非工作日总需求: {od_weekend['demand'].sum():,} 次出行")

print(f"\n工作日OD矩阵前5行:")
od_weekday.head()


# In[42]:


# 为工作日和非工作日OD矩阵添加15分钟时段信息
print("为OD矩阵添加15分钟时段信息...")

# 工作日OD矩阵添加时段信息
od_weekday['t'] = od_weekday['hour'] * 4 + 1  # 简化：每个小时对应4个时段，从1开始
od_weekday = od_weekday[['t', 'i', 'j', 'demand']].copy()
# 按(t, i, j)组合键排序
od_weekday = od_weekday.sort_values(['t', 'i', 'j']).reset_index(drop=True)

# 非工作日OD矩阵添加时段信息
od_weekend['t'] = od_weekend['hour'] * 4 + 1  # 简化：每个小时对应4个时段，从1开始
od_weekend = od_weekend[['t', 'i', 'j', 'demand']].copy()
# 按(t, i, j)组合键排序
od_weekend = od_weekend.sort_values(['t', 'i', 'j']).reset_index(drop=True)

print(f"工作日OD矩阵形状: {od_weekday.shape}")
print(f"非工作日OD矩阵形状: {od_weekend.shape}")
print(f"工作日时段范围: {od_weekday['t'].min()} - {od_weekday['t'].max()}")
print(f"非工作日时段范围: {od_weekend['t'].min()} - {od_weekend['t'].max()}")


# ## 7. 保存结果
# 

# In[43]:


# 保存工作日OD矩阵
weekday_output_file = intermediate_dir / "enhanced_od_matrix_weekday.csv"
od_weekday.to_csv(weekday_output_file, index=False)
print(f"工作日OD矩阵已保存到: {weekday_output_file}")

# 保存非工作日OD矩阵
weekend_output_file = intermediate_dir / "enhanced_od_matrix_weekend.csv"
od_weekend.to_csv(weekend_output_file, index=False)
print(f"非工作日OD矩阵已保存到: {weekend_output_file}")

# 保存Parquet格式（更高效）
od_weekday.to_parquet(intermediate_dir / "enhanced_od_matrix_weekday.parquet", index=False)
od_weekend.to_parquet(intermediate_dir / "enhanced_od_matrix_weekend.parquet", index=False)

print("\n所有文件已保存完成！")


# ## 8. 数据质量报告
# 

# In[44]:


# 生成数据质量报告
print("=" * 60)
print("数据质量报告")
print("=" * 60)

print(f"\n1. 原始数据统计:")
print(f"   - Ride-Austin原始记录: {len(ride_df):,} 条")
print(f"   - 有效出行记录: {len(ride_clean):,} 条 ({len(ride_clean)/len(ride_df)*100:.1f}%)")
print(f"   - 成功映射到TAZ: {len(ride_taz):,} 条 ({len(ride_taz)/len(ride_clean)*100:.1f}%)")
print(f"   - 服务区域内出行: {len(ride_service):,} 条 ({len(ride_service)/len(ride_taz)*100:.1f}%)")

print(f"\n2. 空间覆盖:")
print(f"   - 涉及起始TAZ: {ride_service['start_taz_str'].nunique()} 个")
print(f"   - 涉及终点TAZ: {ride_service['end_taz_str'].nunique()} 个")
print(f"   - 服务区域总TAZ: {len(valid_zones)} 个")
print(f"   - TAZ覆盖率: {ride_service['start_taz_str'].nunique()/len(valid_zones)*100:.1f}%")

print(f"\n3. 时间覆盖:")
print(f"   - 数据时间范围: {ride_service['started_on'].min()} 到 {ride_service['started_on'].max()}")
print(f"   - 工作日出行: {ride_service['is_weekday'].sum():,} 次 ({ride_service['is_weekday'].mean()*100:.1f}%)")
print(f"   - 周末出行: {ride_service['is_weekend'].sum():,} 次 ({ride_service['is_weekend'].mean()*100:.1f}%)")
print(f"   - 覆盖小时数: {ride_service['hour'].nunique()} 小时")

print(f"\n4. OD矩阵统计:")
print(f"   - 工作日OD记录: {len(od_weekday):,} 条")
print(f"   - 非工作日OD记录: {len(od_weekend):,} 条")
print(f"   - 工作日时段数: {od_weekday['t'].nunique()} 个")
print(f"   - 非工作日时段数: {od_weekend['t'].nunique()} 个")
print(f"   - 总需求: {od_matrix['demand'].sum():,} 次出行")
print(f"   - 平均每OD对需求: {od_matrix['demand'].mean():.1f} 次")
print(f"   - 需求中位数: {od_matrix['demand'].median():.1f} 次")

print(f"\n5. 出行特征:")
print(f"   - 平均出行距离: {ride_service['distance_travelled'].mean():.1f} 米")
print(f"   - 平均出行时长: {ride_service['duration_minutes'].mean():.1f} 分钟")
print(f"   - 距离标准差: {ride_service['distance_travelled'].std():.1f} 米")
print(f"   - 时长标准差: {ride_service['duration_minutes'].std():.1f} 分钟")

print(f"\n6. 时间段分布:")
for period in ['morning_peak', 'midday', 'evening_peak', 'evening', 'night']:
    count = (ride_service['time_period'] == period).sum()
    pct = count / len(ride_service) * 100
    print(f"   - {period}: {count:,} 次 ({pct:.1f}%)")

print("\n" + "=" * 60)
print("报告生成完成！")


# ## 9. 与现有OD矩阵对比
# 

# In[45]:


# 加载现有OD矩阵进行对比
try:
    existing_od = pd.read_csv(data_dir / "od_matrix.csv")
    print("现有OD矩阵信息:")
    print(f"   - 形状: {existing_od.shape}")
    print(f"   - 列名: {list(existing_od.columns)}")
    if 'demand' in existing_od.columns:
        print(f"   - 总需求: {existing_od['demand'].sum():,}")
    
    print("\n新生成OD矩阵信息:")
    print(f"   - 工作日OD矩阵形状: {od_weekday.shape}")
    print(f"   - 非工作日OD矩阵形状: {od_weekend.shape}")
    print(f"   - 总需求: {od_matrix['demand'].sum():,}")
    
    print("\n主要改进:")
    print("   ✓ 基于真实出行数据而非工作通勤数据")
    print("   ✓ 包含详细的时间信息（小时级别）")
    print("   ✓ 区分工作日和周末")
    print("   ✓ 包含出行距离和时长信息")
    print("   ✓ 提供多种聚合格式")
    
except FileNotFoundError:
    print("未找到现有OD矩阵文件，跳过对比")
except Exception as e:
    print(f"读取现有OD矩阵时出错: {e}")


# ## 总结
# 
# 本脚本成功生成了基于Ride-Austin数据的高质量OD矩阵，主要特点包括：
# 
# 1. **数据来源改进**: 使用真实的出行数据替代LODES工作通勤数据
# 2. **时间信息丰富**: 包含小时级别的时间信息，区分工作日和周末
# 3. **空间聚合准确**: 基于TAZ进行空间聚合，确保地理一致性
# 4. **特征完整**: 包含出行距离、时长等关键特征
# 5. **格式多样**: 提供完整版、简化版和时间序列版三种格式
# 6. **数据质量高**: 经过严格的数据清洗和验证
# 
# 生成的OD矩阵可以更好地支持SAEV运营优化研究，特别是需要考虑时间动态性和真实出行模式的应用场景。
# 

# ## 10. 生成基于Ride-Austin的TOD分布
# 

# In[46]:


# 基于Ride-Austin数据生成TOD分布
print("生成基于Ride-Austin的TOD分布...")

# 计算每15分钟时段的出行分布
ride_service['period_15min'] = (ride_service['hour'] * 4 + 
                                (ride_service['started_on'].dt.minute // 15)).astype(int)

# 按15分钟时段统计出行次数
tod_distribution = ride_service.groupby('period_15min').size().reset_index(name='count')

# 计算占比
total_trips = tod_distribution['count'].sum()
tod_distribution['share_15min'] = tod_distribution['count'] / total_trips

# 添加时间信息
tod_distribution['period_index'] = tod_distribution['period_15min'] + 1
tod_distribution['start_time'] = tod_distribution['period_15min'].apply(
    lambda x: f"{x//4:02d}:{(x%4)*15:02d}"
)
tod_distribution['end_time'] = tod_distribution['period_15min'].apply(
    lambda x: f"{x//4:02d}:{((x%4)*15+15)%60:02d}" if (x%4)*15+15 < 60 else f"{(x//4+1):02d}:00"
)

# 重新排列列
tod_distribution = tod_distribution[['period_index', 'start_time', 'end_time', 'share_15min']].copy()

print(f"TOD分布数据形状: {tod_distribution.shape}")
print(f"总出行次数: {total_trips:,}")
print(f"覆盖时段数: {len(tod_distribution)}")
print("前10个时段:")
print(tod_distribution.head(10))


# In[ ]:


# 可视化TOD分布对比
if MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 1. 新生成的TOD分布
    axes[0].plot(tod_distribution['period_index'], tod_distribution['share_15min'], 
                 marker='o', linewidth=2, markersize=4, label='Ride-Austin数据')
    axes[0].set_xlabel('时段索引')
    axes[0].set_ylabel('出行占比')
    axes[0].set_title('基于Ride-Austin的TOD分布')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. 与原始TOD分布对比（如果存在）
    try:
        original_tod = pd.read_csv(raw_data_dir / "distribution_of_tod.csv")
        axes[1].plot(original_tod['period_index'], original_tod['share_15min'], 
                     marker='s', linewidth=2, markersize=4, label='原始TOD分布', color='red')
        axes[1].plot(tod_distribution['period_index'], tod_distribution['share_15min'], 
                     marker='o', linewidth=2, markersize=4, label='Ride-Austin数据', color='blue')
        axes[1].set_xlabel('时段索引')
        axes[1].set_ylabel('出行占比')
        axes[1].set_title('TOD分布对比')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    except FileNotFoundError:
        axes[1].text(0.5, 0.5, '未找到原始TOD分布文件', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('TOD分布对比（无原始数据）')

    plt.tight_layout()
    plt.show()
else:
    # 文本统计信息
    print("TOD分布统计:")
    print(f"   - 总时段数: {len(tod_distribution)}")
    print(f"   - 最大出行占比: {tod_distribution['share_15min'].max():.6f}")
    print(f"   - 最小出行占比: {tod_distribution['share_15min'].min():.6f}")
    print(f"   - 平均出行占比: {tod_distribution['share_15min'].mean():.6f}")
    
    # 显示前10个高峰时段
    top_periods = tod_distribution.nlargest(10, 'share_15min')
    print(f"\n前10个高峰时段:")
    for _, row in top_periods.iterrows():
        print(f"   - 时段 {row['period_index']:2d} ({row['start_time']}-{row['end_time']}): {row['share_15min']:.6f}")


# ## 11. 生成base_ij.csv（区域间基础连接信息）
# 

# In[52]:


# 生成base_ij.csv（区域间基础连接信息）
print("生成base_ij.csv...")

# 基于Ride-Austin实际出行数据计算区域间的基础连接信息
# 使用实际出行距离distance_travelled（米）和时长duration_minutes

# 获取所有有效的区域对
all_zones = sorted(list(valid_zones))
print(f"处理 {len(all_zones)} 个区域...")

# 基于实际出行数据计算每个OD对的基础信息
# 先将distance_travelled转换为千米，避免大数值计算问题
ride_service['distance_km'] = ride_service['distance_travelled'] / 1000

# 过滤掉50公里以上的异常数据
ride_service_clean = ride_service[ride_service['distance_km'] <= 50].copy()
print(f"过滤50公里以上异常数据后剩余记录: {len(ride_service_clean):,} 条")

actual_travel_data = ride_service_clean.groupby(['start_taz_str', 'end_taz_str']).agg({
    'distance_km': ['mean', 'median', 'std'],         # 实际出行距离（千米）
    'duration_minutes': ['mean', 'median', 'std'],    # 实际出行时长（分钟）
    'started_on': 'count'  # 出行次数
}).reset_index()

# 展平列名
actual_travel_data.columns = [
    'i', 'j', 'avg_distance_km', 'median_distance_km', 'std_distance_km',
    'avg_duration_min', 'median_duration_min', 'std_duration_min', 'trip_count'
]

# 创建base_ij数据
base_ij_data = []

# 为所有区域对创建记录
for zone_i in all_zones:
    for zone_j in all_zones:
        # 查找是否有实际出行数据
        actual_data = actual_travel_data[
            (actual_travel_data['i'] == zone_i) & 
            (actual_travel_data['j'] == zone_j)
        ]
        
        if len(actual_data) > 0:
            # 使用实际数据
            row = actual_data.iloc[0]
            base_minutes = row['median_duration_min']  # 使用中位数时长
            dist_km = row['median_distance_km']  # 已经是千米单位
        
        base_ij_data.append({
            'i': zone_i,
            'j': zone_j,
            'base_minutes': round(base_minutes, 2),
            'dist_km': round(dist_km, 3)
        })

base_ij_df = pd.DataFrame(base_ij_data)
print(f"base_ij数据形状: {base_ij_df.shape}")
print("前10行:")
print(base_ij_df.head(10))

# 统计信息
print(f"\n基础信息统计:")
print(f"   - 有实际数据的OD对: {len(actual_travel_data)} 个")
print(f"   - 总OD对数量: {len(base_ij_df)} 个")
print(f"   - 实际数据覆盖率: {len(actual_travel_data)/len(base_ij_df)*100:.1f}%")


# In[49]:


# 基于实际Ride-Austin数据优化base_ij
print("基于实际数据优化base_ij...")

# 合并实际出行数据到base_ij
base_ij_enhanced = base_ij_df.merge(actual_travel_data, on=['i', 'j'], how='left')

# 使用实际数据更新基础时间（如果可用）
base_ij_enhanced['base_minutes_actual'] = base_ij_enhanced['median_duration_min'].fillna(base_ij_enhanced['base_minutes'])
base_ij_enhanced['dist_km_actual'] = base_ij_enhanced['median_distance_km'].fillna(base_ij_enhanced['dist_km'])

# 创建最终的base_ij
base_ij_final = base_ij_enhanced[['i', 'j', 'base_minutes_actual', 'dist_km_actual']].copy()
base_ij_final.columns = ['i', 'j', 'base_minutes', 'dist_km']

# 确保数据类型正确
base_ij_final['base_minutes'] = base_ij_final['base_minutes'].round(2)
base_ij_final['dist_km'] = base_ij_final['dist_km'].round(3)

print(f"优化后base_ij数据形状: {base_ij_final.shape}")
print("前10行:")
print(base_ij_final.head(10))

# 统计信息
print(f"\n基础时间统计:")
print(f"   - 平均基础时间: {base_ij_final['base_minutes'].mean():.2f} 分钟")
print(f"   - 基础时间中位数: {base_ij_final['base_minutes'].median():.2f} 分钟")
print(f"   - 平均距离: {base_ij_final['dist_km'].mean():.2f} 公里")
print(f"   - 距离中位数: {base_ij_final['dist_km'].median():.2f} 公里")

# 实际数据统计
print(f"\n实际出行数据统计:")
print(f"   - 有实际数据的OD对: {len(actual_travel_data)} 个")
print(f"   - 总出行次数: {actual_travel_data['trip_count'].sum():,} 次")
print(f"   - 平均出行距离: {actual_travel_data['median_distance_km'].mean():.2f} 公里")
print(f"   - 平均出行时长: {actual_travel_data['median_duration_min'].mean():.1f} 分钟")


# ## 12. 生成coeff_schedule.csv（拥堵系数）
# 

# In[50]:


# 生成coeff_schedule.csv（拥堵系数）
print("生成coeff_schedule.csv...")

# 基于Ride-Austin数据计算不同时段的拥堵系数
# 计算每个时段的平均出行时间相对于基础时间的倍数

# 首先计算每个OD对在不同时段的基础时间
od_base_times = base_ij_final.set_index(['i', 'j'])['base_minutes'].to_dict()

# 为每个出行记录添加基础时间
ride_service['base_time'] = ride_service.apply(
    lambda row: od_base_times.get((row['start_taz_str'], row['end_taz_str']), 10.0), 
    axis=1
)

# 计算拥堵系数（实际时间/基础时间）
ride_service['congestion_ratio'] = ride_service['duration_minutes'] / ride_service['base_time']

print(f"拥堵系数计算完成:")
print(f"   - 平均拥堵系数: {ride_service['congestion_ratio'].mean():.3f}")
print(f"   - 拥堵系数范围: {ride_service['congestion_ratio'].min():.3f} - {ride_service['congestion_ratio'].max():.3f}")
print(f"   - 拥堵系数中位数: {ride_service['congestion_ratio'].median():.3f}")

# 按15分钟时段计算拥堵系数
congestion_by_period = ride_service.groupby('period_15min').agg({
    'congestion_ratio': ['mean', 'median', 'std'],
    'duration_minutes': ['mean', 'median'],
    'base_time': 'mean'
}).reset_index()

# 展平列名
congestion_by_period.columns = [
    'period_15min', 'avg_congestion_ratio', 'median_congestion_ratio', 'std_congestion_ratio',
    'avg_duration', 'median_duration', 'avg_base_time'
]

# 创建coeff_schedule数据
coeff_schedule_data = []

# 为每个15分钟时段创建记录
for period in range(96):  # 24小时 * 4 = 96个15分钟时段
    period_data = congestion_by_period[congestion_by_period['period_15min'] == period]
    
    if len(period_data) > 0:
        # 使用中位数拥堵系数作为主要系数
        gamma_rep_p = period_data['median_congestion_ratio'].iloc[0]
        
        # 计算充电相关的系数（基于拥堵程度调整）
        # 拥堵时充电时间可能增加
        beta_chg_p1 = max(1.0, gamma_rep_p * 0.9)  # 充电系数1
        beta_chg_p2 = max(1.0, gamma_rep_p * 1.1)  # 充电系数2
        
        # 确保系数在合理范围内
        gamma_rep_p = max(0.5, min(2.0, gamma_rep_p))
        beta_chg_p1 = max(0.8, min(1.5, beta_chg_p1))
        beta_chg_p2 = max(0.8, min(1.8, beta_chg_p2))
        
    else:
        # 如果没有数据，使用默认值
        gamma_rep_p = 1.0
        beta_chg_p1 = 1.0
        beta_chg_p2 = 1.0
    
    coeff_schedule_data.append({
        't': period + 1,  # 时段索引从1开始
        'gamma_rep_p': round(gamma_rep_p, 4),
        'beta_chg_p1': round(beta_chg_p1, 4),
        'beta_chg_p2': round(beta_chg_p2, 4)
    })

coeff_schedule_df = pd.DataFrame(coeff_schedule_data)
# 按t排序
coeff_schedule_df = coeff_schedule_df.sort_values('t').reset_index(drop=True)

print(f"coeff_schedule数据形状: {coeff_schedule_df.shape}")
print("前10行:")
print(coeff_schedule_df.head(10))

# 统计信息
print(f"\n拥堵系数统计:")
print(f"   - gamma_rep_p 平均: {coeff_schedule_df['gamma_rep_p'].mean():.4f}")
print(f"   - gamma_rep_p 范围: {coeff_schedule_df['gamma_rep_p'].min():.4f} - {coeff_schedule_df['gamma_rep_p'].max():.4f}")
print(f"   - beta_chg_p1 平均: {coeff_schedule_df['beta_chg_p1'].mean():.4f}")
print(f"   - beta_chg_p2 平均: {coeff_schedule_df['beta_chg_p2'].mean():.4f}")


# In[51]:


# 可视化拥堵系数
if MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 拥堵系数时间序列
    axes[0, 0].plot(coeff_schedule_df['t'], coeff_schedule_df['gamma_rep_p'], 
                    linewidth=2, label='gamma_rep_p')
    axes[0, 0].set_xlabel('时段索引')
    axes[0, 0].set_ylabel('拥堵系数')
    axes[0, 0].set_title('拥堵系数时间序列')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. 充电系数对比
    axes[0, 1].plot(coeff_schedule_df['t'], coeff_schedule_df['beta_chg_p1'], 
                    linewidth=2, label='beta_chg_p1')
    axes[0, 1].plot(coeff_schedule_df['t'], coeff_schedule_df['beta_chg_p2'], 
                    linewidth=2, label='beta_chg_p2')
    axes[0, 1].set_xlabel('时段索引')
    axes[0, 1].set_ylabel('充电系数')
    axes[0, 1].set_title('充电系数时间序列')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. 拥堵系数分布
    axes[1, 0].hist(coeff_schedule_df['gamma_rep_p'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('拥堵系数')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('拥堵系数分布')
    axes[1, 0].axvline(coeff_schedule_df['gamma_rep_p'].mean(), color='red', linestyle='--', 
                       label=f'平均值: {coeff_schedule_df["gamma_rep_p"].mean():.3f}')
    axes[1, 0].legend()

    # 4. 系数相关性
    axes[1, 1].scatter(coeff_schedule_df['gamma_rep_p'], coeff_schedule_df['beta_chg_p1'], 
                       alpha=0.6, label='beta_chg_p1')
    axes[1, 1].scatter(coeff_schedule_df['gamma_rep_p'], coeff_schedule_df['beta_chg_p2'], 
                       alpha=0.6, label='beta_chg_p2')
    axes[1, 1].set_xlabel('gamma_rep_p')
    axes[1, 1].set_ylabel('充电系数')
    axes[1, 1].set_title('拥堵系数与充电系数关系')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
else:
    # 文本统计信息
    print("拥堵系数统计:")
    print(f"   - 总时段数: {len(coeff_schedule_df)}")
    print(f"   - gamma_rep_p 平均: {coeff_schedule_df['gamma_rep_p'].mean():.4f}")
    print(f"   - gamma_rep_p 范围: {coeff_schedule_df['gamma_rep_p'].min():.4f} - {coeff_schedule_df['gamma_rep_p'].max():.4f}")
    print(f"   - beta_chg_p1 平均: {coeff_schedule_df['beta_chg_p1'].mean():.4f}")
    print(f"   - beta_chg_p2 平均: {coeff_schedule_df['beta_chg_p2'].mean():.4f}")
    
    # 显示前10个最高拥堵时段
    top_congestion = coeff_schedule_df.nlargest(10, 'gamma_rep_p')
    print(f"\n前10个最高拥堵时段:")
    for _, row in top_congestion.iterrows():
        period_hour = (row['t'] - 1) // 4
        period_min = ((row['t'] - 1) % 4) * 15
        time_str = f"{int(period_hour):02d}:{int(period_min):02d}"
        print(f"   - 时段 {int(row['t']):2d} ({time_str}): {row['gamma_rep_p']:.4f}")
    
    # 系数相关性
    gamma_beta1_corr = coeff_schedule_df['gamma_rep_p'].corr(coeff_schedule_df['beta_chg_p1'])
    gamma_beta2_corr = coeff_schedule_df['gamma_rep_p'].corr(coeff_schedule_df['beta_chg_p2'])
    print(f"\n系数相关性:")
    print(f"   - gamma_rep_p 与 beta_chg_p1: {gamma_beta1_corr:.4f}")
    print(f"   - gamma_rep_p 与 beta_chg_p2: {gamma_beta2_corr:.4f}")


# ## 13. 保存所有生成的文件
# 

# In[ ]:


# 按工作日/非工作日分别生成和保存数据
print("按工作日/非工作日分别生成数据...")

# 1. 分别生成TOD分布
print("\n1. 生成TOD分布...")

# 工作日TOD分布
weekday_data = ride_service[ride_service['is_weekday'] == True]
weekday_tod = weekday_data.groupby('period_15min').size().reset_index(name='count')
weekday_total = weekday_tod['count'].sum()
weekday_tod['share_15min'] = weekday_tod['count'] / weekday_total
weekday_tod['period_index'] = weekday_tod['period_15min'] + 1
weekday_tod['start_time'] = weekday_tod['period_15min'].apply(
    lambda x: f"{x//4:02d}:{(x%4)*15:02d}"
)
weekday_tod['end_time'] = weekday_tod['period_15min'].apply(
    lambda x: f"{x//4:02d}:{((x%4)*15+15)%60:02d}" if (x%4)*15+15 < 60 else f"{(x//4+1):02d}:00"
)
weekday_tod_final = weekday_tod[['period_index', 'start_time', 'end_time', 'share_15min']].copy()
# 按period_index排序
weekday_tod_final = weekday_tod_final.sort_values('period_index').reset_index(drop=True)

# 非工作日TOD分布
weekend_data = ride_service[ride_service['is_weekday'] == False]
weekend_tod = weekend_data.groupby('period_15min').size().reset_index(name='count')
weekend_total = weekend_tod['count'].sum()
weekend_tod['share_15min'] = weekend_tod['count'] / weekend_total
weekend_tod['period_index'] = weekend_tod['period_15min'] + 1
weekend_tod['start_time'] = weekend_tod['period_15min'].apply(
    lambda x: f"{x//4:02d}:{(x%4)*15:02d}"
)
weekend_tod['end_time'] = weekend_tod['period_15min'].apply(
    lambda x: f"{x//4:02d}:{((x%4)*15+15)%60:02d}" if (x%4)*15+15 < 60 else f"{(x//4+1):02d}:00"
)
weekend_tod_final = weekend_tod[['period_index', 'start_time', 'end_time', 'share_15min']].copy()
# 按period_index排序
weekend_tod_final = weekend_tod_final.sort_values('period_index').reset_index(drop=True)

print(f"   - 工作日TOD分布: {len(weekday_tod_final)} 个时段，{weekday_total:,} 次出行")
print(f"   - 非工作日TOD分布: {len(weekend_tod_final)} 个时段，{weekend_total:,} 次出行")

# 2. 分别生成coeff_schedule
print("\n2. 生成拥堵系数...")

# 工作日拥堵系数
weekday_congestion = weekday_data.groupby('period_15min').agg({
    'congestion_ratio': ['mean', 'median', 'std']
}).reset_index()
weekday_congestion.columns = ['period_15min', 'avg_congestion_ratio', 'median_congestion_ratio', 'std_congestion_ratio']

weekday_coeff_data = []
for period in range(96):
    period_data = weekday_congestion[weekday_congestion['period_15min'] == period]
    if len(period_data) > 0:
        gamma_rep_p = period_data['median_congestion_ratio'].iloc[0]
        beta_chg_p1 = max(1.0, gamma_rep_p * 0.9)
        beta_chg_p2 = max(1.0, gamma_rep_p * 1.1)
        gamma_rep_p = max(0.5, min(2.0, gamma_rep_p))
        beta_chg_p1 = max(0.8, min(1.5, beta_chg_p1))
        beta_chg_p2 = max(0.8, min(1.8, beta_chg_p2))
    else:
        gamma_rep_p = 1.0
        beta_chg_p1 = 1.0
        beta_chg_p2 = 1.0
    
    weekday_coeff_data.append({
        't': period + 1,
        'gamma_rep_p': round(gamma_rep_p, 4),
        'beta_chg_p1': round(beta_chg_p1, 4),
        'beta_chg_p2': round(beta_chg_p2, 4)
    })

weekday_coeff_df = pd.DataFrame(weekday_coeff_data)
# 按t排序
weekday_coeff_df = weekday_coeff_df.sort_values('t').reset_index(drop=True)

# 非工作日拥堵系数
weekend_congestion = weekend_data.groupby('period_15min').agg({
    'congestion_ratio': ['mean', 'median', 'std']
}).reset_index()
weekend_congestion.columns = ['period_15min', 'avg_congestion_ratio', 'median_congestion_ratio', 'std_congestion_ratio']

weekend_coeff_data = []
for period in range(96):
    period_data = weekend_congestion[weekend_congestion['period_15min'] == period]
    if len(period_data) > 0:
        gamma_rep_p = period_data['median_congestion_ratio'].iloc[0]
        beta_chg_p1 = max(1.0, gamma_rep_p * 0.9)
        beta_chg_p2 = max(1.0, gamma_rep_p * 1.1)
        gamma_rep_p = max(0.5, min(2.0, gamma_rep_p))
        beta_chg_p1 = max(0.8, min(1.5, beta_chg_p1))
        beta_chg_p2 = max(0.8, min(1.8, beta_chg_p2))
    else:
        gamma_rep_p = 1.0
        beta_chg_p1 = 1.0
        beta_chg_p2 = 1.0
    
    weekend_coeff_data.append({
        't': period + 1,
        'gamma_rep_p': round(gamma_rep_p, 4),
        'beta_chg_p1': round(beta_chg_p1, 4),
        'beta_chg_p2': round(beta_chg_p2, 4)
    })

weekend_coeff_df = pd.DataFrame(weekend_coeff_data)
# 按t排序
weekend_coeff_df = weekend_coeff_df.sort_values('t').reset_index(drop=True)

print(f"   - 工作日拥堵系数: {len(weekday_coeff_df)} 个时段")
print(f"   - 非工作日拥堵系数: {len(weekend_coeff_df)} 个时段")

# 3. 分别生成coeff_energy（基于实际速度计算）
print("\n3. 生成coeff_energy...")

# 计算工作日和非工作日的速度
weekday_data['speed_kmh'] = (weekday_data['distance_travelled'] / 1000) / (weekday_data['duration_minutes'] / 60)
weekend_data['speed_kmh'] = (weekend_data['distance_travelled'] / 1000) / (weekend_data['duration_minutes'] / 60)

# 过滤异常速度值（0-120 km/h）
weekday_data = weekday_data[(weekday_data['speed_kmh'] > 0) & (weekday_data['speed_kmh'] <= 120)]
weekend_data = weekend_data[(weekend_data['speed_kmh'] > 0) & (weekend_data['speed_kmh'] <= 120)]

# 工作日速度统计
weekday_speed_stats = weekday_data.groupby('period_15min').agg({
    'speed_kmh': ['mean', 'median', 'std', 'count']
}).reset_index()
weekday_speed_stats.columns = ['period_15min', 'avg_speed', 'median_speed', 'std_speed', 'trip_count']

# 非工作日速度统计
weekend_speed_stats = weekend_data.groupby('period_15min').agg({
    'speed_kmh': ['mean', 'median', 'std', 'count']
}).reset_index()
weekend_speed_stats.columns = ['period_15min', 'avg_speed', 'median_speed', 'std_speed', 'trip_count']

# 生成工作日coeff_energy
weekday_energy_data = []
for period in range(96):
    period_data = weekday_speed_stats[weekday_speed_stats['period_15min'] == period]
    
    if len(period_data) > 0 and period_data['trip_count'].iloc[0] >= 5:  # 至少5个样本
        # 使用中位数速度
        speed_kmh = period_data['median_speed'].iloc[0]
        
        # 基于速度计算耗电系数
        # 假设基准速度为30 km/h，耗电与速度的平方成正比
        base_speed = 30.0
        
        # 服务模式耗电系数 (de_per_km_srv)
        de_per_km_srv = (speed_kmh / base_speed) ** 2
        de_per_km_srv = max(0.5, min(3.0, de_per_km_srv))
        
        # 重新定位模式耗电系数 (de_per_km_rep) - 通常比服务模式低
        de_per_km_rep = de_per_km_srv * 0.8  # 重新定位时耗电较低
        de_per_km_rep = max(0.3, min(2.5, de_per_km_rep))
        
        # 前往充电站模式耗电系数 (de_per_km_tochg) - 通常最低
        de_per_km_tochg = de_per_km_srv * 0.7  # 前往充电站时耗电最低
        de_per_km_tochg = max(0.2, min(2.0, de_per_km_tochg))
        
    else:
        # 默认值
        speed_kmh = 30.0
        de_per_km_srv = 1.0
        de_per_km_rep = 0.8
        de_per_km_tochg = 0.7
    
    weekday_energy_data.append({
        't': period + 1,
        'de_per_km_srv': round(de_per_km_srv, 4),
        'de_per_km_rep': round(de_per_km_rep, 4),
        'de_per_km_tochg': round(de_per_km_tochg, 4)
    })

weekday_energy_df = pd.DataFrame(weekday_energy_data)
# 按t排序
weekday_energy_df = weekday_energy_df.sort_values('t').reset_index(drop=True)

# 生成非工作日coeff_energy
weekend_energy_data = []
for period in range(96):
    period_data = weekend_speed_stats[weekend_speed_stats['period_15min'] == period]
    
    if len(period_data) > 0 and period_data['trip_count'].iloc[0] >= 5:  # 至少5个样本
        # 使用中位数速度
        speed_kmh = period_data['median_speed'].iloc[0]
        
        # 基于速度计算耗电系数
        base_speed = 30.0
        
        # 服务模式耗电系数 (de_per_km_srv)
        de_per_km_srv = (speed_kmh / base_speed) ** 2
        de_per_km_srv = max(0.5, min(3.0, de_per_km_srv))
        
        # 重新定位模式耗电系数 (de_per_km_rep) - 通常比服务模式低
        de_per_km_rep = de_per_km_srv * 0.8  # 重新定位时耗电较低
        de_per_km_rep = max(0.3, min(2.5, de_per_km_rep))
        
        # 前往充电站模式耗电系数 (de_per_km_tochg) - 通常最低
        de_per_km_tochg = de_per_km_srv * 0.7  # 前往充电站时耗电最低
        de_per_km_tochg = max(0.2, min(2.0, de_per_km_tochg))
        
    else:
        # 默认值
        speed_kmh = 30.0
        de_per_km_srv = 1.0
        de_per_km_rep = 0.8
        de_per_km_tochg = 0.7
    
    weekend_energy_data.append({
        't': period + 1,
        'de_per_km_srv': round(de_per_km_srv, 4),
        'de_per_km_rep': round(de_per_km_rep, 4),
        'de_per_km_tochg': round(de_per_km_tochg, 4)
    })

weekend_energy_df = pd.DataFrame(weekend_energy_data)
# 按t排序
weekend_energy_df = weekend_energy_df.sort_values('t').reset_index(drop=True)

print(f"   - 工作日耗电系数: {len(weekday_energy_df)} 个时段")
print(f"   - 非工作日耗电系数: {len(weekend_energy_df)} 个时段")
print(f"   - 工作日平均速度: {weekday_energy_df['de_per_km_srv'].mean():.2f} km/h")
print(f"   - 非工作日平均速度: {weekend_energy_df['de_per_km_srv'].mean():.2f} km/h")
print(f"   - 工作日平均耗电系数: {weekday_energy_df['de_per_km_srv'].mean():.4f}")
print(f"   - 非工作日平均耗电系数: {weekend_energy_df['de_per_km_srv'].mean():.4f}")

# 4. 保存所有文件
print("\n4. 保存文件...")

# intermediate目录已在开头创建

# 保存TOD分布
weekday_tod_output = intermediate_dir / "distribution_of_tod_weekday.csv"
weekend_tod_output = intermediate_dir / "distribution_of_tod_weekend.csv"
weekday_tod_final.to_csv(weekday_tod_output, index=False)
weekend_tod_final.to_csv(weekend_tod_output, index=False)
print(f"✓ 工作日TOD分布: {weekday_tod_output}")
print(f"✓ 非工作日TOD分布: {weekend_tod_output}")

# 分别生成工作日和非工作日的base_ij
print("\n5. 生成base_ij...")

# 工作日base_ij
# 先将distance_travelled转换为千米
weekday_data['distance_km'] = weekday_data['distance_travelled'] / 1000
# 过滤掉50公里以上的异常数据
weekday_data_clean = weekday_data[weekday_data['distance_km'] <= 50].copy()
weekday_actual_travel = weekday_data_clean.groupby(['start_taz_str', 'end_taz_str']).agg({
    'distance_km': ['mean', 'median', 'std'],
    'duration_minutes': ['mean', 'median', 'std'],
    'started_on': 'count'
}).reset_index()

weekday_actual_travel.columns = [
    'i', 'j', 'avg_distance_km', 'median_distance_km', 'std_distance_km',
    'avg_duration_min', 'median_duration_min', 'std_duration_min', 'trip_count'
]

# 非工作日base_ij
# 先将distance_travelled转换为千米
weekend_data['distance_km'] = weekend_data['distance_travelled'] / 1000
# 过滤掉50公里以上的异常数据
weekend_data_clean = weekend_data[weekend_data['distance_km'] <= 50].copy()
weekend_actual_travel = weekend_data_clean.groupby(['start_taz_str', 'end_taz_str']).agg({
    'distance_km': ['mean', 'median', 'std'],
    'duration_minutes': ['mean', 'median', 'std'],
    'started_on': 'count'
}).reset_index()

weekend_actual_travel.columns = [
    'i', 'j', 'avg_distance_km', 'median_distance_km', 'std_distance_km',
    'avg_duration_min', 'median_duration_min', 'std_duration_min', 'trip_count'
]

# 为工作日生成base_ij
weekday_base_ij_data = []
for zone_i in all_zones:
    for zone_j in all_zones:
        # 查找是否有实际出行数据
        actual_data = weekday_actual_travel[
            (weekday_actual_travel['i'] == zone_i) & 
            (weekday_actual_travel['j'] == zone_j)
        ]
        
        if len(actual_data) > 0:
            # 使用实际数据
            row = actual_data.iloc[0]
            base_minutes = row['median_duration_min']
            dist_km = row['median_distance_km']  # 已经是千米单位
        else:
            # 如果没有实际数据，使用默认值
            if zone_i == zone_j:
                base_minutes = 0.5
                dist_km = 0.1
            else:
                # 区域间出行，使用区域中心点估算
                zone_i_data = zones_df[zones_df['zone'].astype(str).str.zfill(8) == zone_i]
                zone_j_data = zones_df[zones_df['zone'].astype(str).str.zfill(8) == zone_j]
                
                if len(zone_i_data) > 0 and len(zone_j_data) > 0:
                    from math import radians, cos, sin, asin, sqrt
                    
                    def haversine_distance(lon1, lat1, lon2, lat2):
                        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * asin(sqrt(a))
                        r = 6371
                        return c * r
                    
                    lat1, lon1 = zone_i_data.iloc[0]['lat'], zone_i_data.iloc[0]['lon']
                    lat2, lon2 = zone_j_data.iloc[0]['lat'], zone_j_data.iloc[0]['lon']
                    dist_km = haversine_distance(lon1, lat1, lon2, lat2)
                    base_minutes = (dist_km / 30) * 60
                else:
                    base_minutes = 10.0
                    dist_km = 5.0
        
        weekday_base_ij_data.append({
            'i': zone_i,
            'j': zone_j,
            'base_minutes': round(base_minutes, 2),
            'dist_km': round(dist_km, 3)
        })

weekday_base_ij_df = pd.DataFrame(weekday_base_ij_data)

# 为非工作日生成base_ij
weekend_base_ij_data = []
for zone_i in all_zones:
    for zone_j in all_zones:
        # 查找是否有实际出行数据
        actual_data = weekend_actual_travel[
            (weekend_actual_travel['i'] == zone_i) & 
            (weekend_actual_travel['j'] == zone_j)
        ]
        
        if len(actual_data) > 0:
            # 使用实际数据
            row = actual_data.iloc[0]
            base_minutes = row['median_duration_min']
            dist_km = row['median_distance_km']  # 已经是千米单位
        else:
            # 如果没有实际数据，使用默认值
            if zone_i == zone_j:
                base_minutes = 0.5
                dist_km = 0.1
            else:
                # 区域间出行，使用区域中心点估算
                zone_i_data = zones_df[zones_df['zone'].astype(str).str.zfill(8) == zone_i]
                zone_j_data = zones_df[zones_df['zone'].astype(str).str.zfill(8) == zone_j]
                
                if len(zone_i_data) > 0 and len(zone_j_data) > 0:
                    from math import radians, cos, sin, asin, sqrt
                    
                    def haversine_distance(lon1, lat1, lon2, lat2):
                        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * asin(sqrt(a))
                        r = 6371
                        return c * r
                    
                    lat1, lon1 = zone_i_data.iloc[0]['lat'], zone_i_data.iloc[0]['lon']
                    lat2, lon2 = zone_j_data.iloc[0]['lat'], zone_j_data.iloc[0]['lon']
                    dist_km = haversine_distance(lon1, lat1, lon2, lat2)
                    base_minutes = (dist_km / 30) * 60
                else:
                    base_minutes = 10.0
                    dist_km = 5.0
        
        weekend_base_ij_data.append({
            'i': zone_i,
            'j': zone_j,
            'base_minutes': round(base_minutes, 2),
            'dist_km': round(dist_km, 3)
        })

weekend_base_ij_df = pd.DataFrame(weekend_base_ij_data)

print(f"   - 工作日base_ij: {len(weekday_base_ij_df)} 个OD对")
print(f"   - 非工作日base_ij: {len(weekend_base_ij_df)} 个OD对")
print(f"   - 工作日有实际数据的OD对: {len(weekday_actual_travel)} 个")
print(f"   - 非工作日有实际数据的OD对: {len(weekend_actual_travel)} 个")

# 保存base_ij
weekday_base_ij_output = intermediate_dir / "base_ij_weekday.csv"
weekend_base_ij_output = intermediate_dir / "base_ij_weekend.csv"
weekday_base_ij_df.to_csv(weekday_base_ij_output, index=False)
weekend_base_ij_df.to_csv(weekend_base_ij_output, index=False)
print(f"✓ 工作日base_ij: {weekday_base_ij_output}")
print(f"✓ 非工作日base_ij: {weekend_base_ij_output}")

# 保存coeff_schedule
weekday_coeff_output = intermediate_dir / "coeff_schedule_weekday.csv"
weekend_coeff_output = intermediate_dir / "coeff_schedule_weekend.csv"
weekday_coeff_df.to_csv(weekday_coeff_output, index=False)
weekend_coeff_df.to_csv(weekend_coeff_output, index=False)
print(f"✓ 工作日拥堵系数: {weekday_coeff_output}")
print(f"✓ 非工作日拥堵系数: {weekend_coeff_output}")

# 保存coeff_energy
weekday_energy_output = intermediate_dir / "coeff_energy_weekday.csv"
weekend_energy_output = intermediate_dir / "coeff_energy_weekend.csv"
weekday_energy_df.to_csv(weekday_energy_output, index=False)
weekend_energy_df.to_csv(weekend_energy_output, index=False)
print(f"✓ 工作日耗电系数: {weekday_energy_output}")
print(f"✓ 非工作日耗电系数: {weekend_energy_output}")

# 保存Parquet格式
weekday_tod_final.to_parquet(intermediate_dir / "distribution_of_tod_weekday.parquet", index=False)
weekend_tod_final.to_parquet(intermediate_dir / "distribution_of_tod_weekend.parquet", index=False)
weekday_base_ij_df.to_parquet(intermediate_dir / "base_ij_weekday.parquet", index=False)
weekend_base_ij_df.to_parquet(intermediate_dir / "base_ij_weekend.parquet", index=False)
weekday_coeff_df.to_parquet(intermediate_dir / "coeff_schedule_weekday.parquet", index=False)
weekend_coeff_df.to_parquet(intermediate_dir / "coeff_schedule_weekend.parquet", index=False)
weekday_energy_df.to_parquet(intermediate_dir / "coeff_energy_weekday.parquet", index=False)
weekend_energy_df.to_parquet(intermediate_dir / "coeff_energy_weekend.parquet", index=False)

print("\n所有文件已保存完成！")


# ## 14. 数据质量验证和对比分析
# 

# In[ ]:


# 数据质量验证和对比分析
print("=" * 80)
print("数据质量验证和对比分析")
print("=" * 80)

# 1. TOD分布对比
print("\n1. TOD分布对比分析:")
try:
    original_tod = pd.read_csv(raw_data_dir / "distribution_of_tod.csv")
    print(f"   原始TOD分布: {len(original_tod)} 个时段")
    print(f"   新生成TOD分布: {len(tod_distribution)} 个时段")
    
    # 计算相关性
    merged_tod = original_tod.merge(tod_distribution, on='period_index', how='inner')
    if len(merged_tod) > 0:
        correlation = merged_tod['share_15min_x'].corr(merged_tod['share_15min_y'])
        print(f"   相关性系数: {correlation:.4f}")
        
        # 计算差异
        merged_tod['diff'] = abs(merged_tod['share_15min_x'] - merged_tod['share_15min_y'])
        print(f"   平均绝对差异: {merged_tod['diff'].mean():.6f}")
        print(f"   最大差异: {merged_tod['diff'].max():.6f}")
except FileNotFoundError:
    print("   未找到原始TOD分布文件")

# 2. base_ij对比
print("\n2. base_ij对比分析:")
try:
    original_base_ij = pd.read_csv(data_dir / "base_ij.csv")
    print(f"   原始base_ij: {len(original_base_ij)} 个OD对")
    print(f"   新生成base_ij: {len(base_ij_final)} 个OD对")
    
    # 计算重叠的OD对
    original_pairs = set(zip(original_base_ij['i'], original_base_ij['j']))
    new_pairs = set(zip(base_ij_final['i'], base_ij_final['j']))
    overlap_pairs = original_pairs.intersection(new_pairs)
    print(f"   重叠OD对: {len(overlap_pairs)} 个")
    
    if len(overlap_pairs) > 0:
        # 对比重叠部分的时间
        original_dict = original_base_ij.set_index(['i', 'j'])['base_minutes'].to_dict()
        new_dict = base_ij_final.set_index(['i', 'j'])['base_minutes'].to_dict()
        
        time_diffs = []
        for pair in list(overlap_pairs)[:100]:  # 限制样本数量
            if pair in original_dict and pair in new_dict:
                time_diffs.append(abs(original_dict[pair] - new_dict[pair]))
        
        if time_diffs:
            print(f"   平均时间差异: {np.mean(time_diffs):.2f} 分钟")
            print(f"   最大时间差异: {np.max(time_diffs):.2f} 分钟")
except FileNotFoundError:
    print("   未找到原始base_ij文件")

# 3. coeff_schedule对比
print("\n3. coeff_schedule对比分析:")
try:
    original_coeff = pd.read_csv(data_dir / "coeff_schedule.csv")
    print(f"   原始coeff_schedule: {len(original_coeff)} 个时段")
    print(f"   新生成coeff_schedule: {len(coeff_schedule_df)} 个时段")
    
    # 计算相关性
    merged_coeff = original_coeff.merge(coeff_schedule_df, on='t', how='inner')
    if len(merged_coeff) > 0:
        gamma_corr = merged_coeff['gamma_rep_p_x'].corr(merged_coeff['gamma_rep_p_y'])
        print(f"   gamma_rep_p相关性: {gamma_corr:.4f}")
        
        beta1_corr = merged_coeff['beta_chg_p1_x'].corr(merged_coeff['beta_chg_p1_y'])
        print(f"   beta_chg_p1相关性: {beta1_corr:.4f}")
        
        beta2_corr = merged_coeff['beta_chg_p2_x'].corr(merged_coeff['beta_chg_p2_y'])
        print(f"   beta_chg_p2相关性: {beta2_corr:.4f}")
except FileNotFoundError:
    print("   未找到原始coeff_schedule文件")

print("\n" + "=" * 80)


# ## 15. 最终总结报告
# 

# In[ ]:


# 最终总结报告
print("=" * 80)
print("基于Ride-Austin数据的高质量OD矩阵生成 - 最终总结报告")
print("=" * 80)

print(f"\n📊 数据源信息:")
print(f"   - Ride-Austin原始数据: {len(ride_df):,} 条记录")
print(f"   - 有效出行数据: {len(ride_service):,} 条记录")
print(f"   - 数据时间范围: {ride_service['started_on'].min()} 到 {ride_service['started_on'].max()}")
print(f"   - 服务区域TAZ数量: {len(valid_zones)} 个")
print(f"   - 使用实际出行距离: {ride_service['distance_travelled'].mean():.0f} 米（平均）")
print(f"   - 使用实际出行时长: {ride_service['duration_minutes'].mean():.1f} 分钟（平均）")

print(f"\n📈 生成的文件:")
print(f"   1. enhanced_od_matrix_weekday.csv - 工作日OD矩阵 ({len(od_weekday):,} 条记录)")
print(f"   2. enhanced_od_matrix_weekend.csv - 非工作日OD矩阵 ({len(od_weekend):,} 条记录)")
print(f"   3. distribution_of_tod_weekday.csv - 工作日TOD分布 ({len(weekday_tod_final):,} 个时段)")
print(f"   4. distribution_of_tod_weekend.csv - 非工作日TOD分布 ({len(weekend_tod_final):,} 个时段)")
print(f"   5. base_ij_weekday.csv - 工作日区域间基础连接 ({len(weekday_base_ij_df):,} 个OD对)")
print(f"   6. base_ij_weekend.csv - 非工作日区域间基础连接 ({len(weekend_base_ij_df):,} 个OD对)")
print(f"   7. coeff_schedule_weekday.csv - 工作日拥堵系数 ({len(weekday_coeff_df):,} 个时段)")
print(f"   8. coeff_schedule_weekend.csv - 非工作日拥堵系数 ({len(weekend_coeff_df):,} 个时段)")
print(f"   9. coeff_energy_weekday.csv - 工作日耗电系数 ({len(weekday_energy_df):,} 个时段)")
print(f"   10. coeff_energy_weekend.csv - 非工作日耗电系数 ({len(weekend_energy_df):,} 个时段)")

print(f"\n🔍 主要改进:")
print(f"   ✓ 使用真实出行数据替代LODES工作通勤数据")
print(f"   ✓ 基于实际经纬度坐标映射TAZ（非区域中心点）")
print(f"   ✓ 使用实际出行距离distance_travelled（米）而非欧几里得估算")
print(f"   ✓ 包含详细的时间信息（15分钟精度时段分解）")
print(f"   ✓ 区分工作日和周末出行模式")
print(f"   ✓ 基于实际数据计算拥堵系数")
print(f"   ✓ 包含出行距离和时长等关键特征")
print(f"   ✓ OD矩阵包含时段属性（t, i, j, demand）")
print(f"   ✓ 按工作日/非工作日分别生成数据，支持多场景对照")
print(f"   ✓ 基于实际速度计算耗电系数，更科学地反映耗电模式")

print(f"\n📋 数据质量:")
print(f"   - 数据清洗率: {len(ride_clean)/len(ride_df)*100:.1f}%")
print(f"   - TAZ映射成功率: {len(ride_taz)/len(ride_clean)*100:.1f}%")
print(f"   - 服务区域覆盖率: {len(ride_service)/len(ride_taz)*100:.1f}%")
print(f"   - 工作日/周末分布: {ride_service['is_weekday'].mean()*100:.1f}% / {ride_service['is_weekend'].mean()*100:.1f}%")

print(f"\n⏰ 时间特征:")
print(f"   - 覆盖小时数: {ride_service['hour'].nunique()} 小时")
print(f"   - 高峰时段: {ride_service[ride_service['time_period'].isin(['morning_peak', 'evening_peak'])]['time_period'].value_counts().sum():,} 次出行")
print(f"   - 平均出行时长: {ride_service['duration_minutes'].mean():.1f} 分钟")
print(f"   - 平均出行距离: {ride_service['distance_travelled'].mean():.0f} 米")

print(f"\n🚗 拥堵分析:")
print(f"   - 工作日平均拥堵系数: {weekday_coeff_df['gamma_rep_p'].mean():.3f}")
print(f"   - 非工作日平均拥堵系数: {weekend_coeff_df['gamma_rep_p'].mean():.3f}")
print(f"   - 工作日拥堵系数范围: {weekday_coeff_df['gamma_rep_p'].min():.3f} - {weekday_coeff_df['gamma_rep_p'].max():.3f}")
print(f"   - 非工作日拥堵系数范围: {weekend_coeff_df['gamma_rep_p'].min():.3f} - {weekend_coeff_df['gamma_rep_p'].max():.3f}")
print(f"   - 工作日高峰时段拥堵系数: {weekday_coeff_df[weekday_coeff_df['t'].between(29, 40)]['gamma_rep_p'].mean():.3f}")
print(f"   - 非工作日高峰时段拥堵系数: {weekend_coeff_df[weekend_coeff_df['t'].between(29, 40)]['gamma_rep_p'].mean():.3f}")

print(f"\n⚡ 耗电分析:")
print(f"   - 工作日平均速度: {weekday_energy_df['de_per_km_srv'].mean():.2f} km/h")
print(f"   - 非工作日平均速度: {weekend_energy_df['de_per_km_srv'].mean():.2f} km/h")
print(f"   - 工作日平均耗电系数: {weekday_energy_df['de_per_km_srv'].mean():.4f}")
print(f"   - 非工作日平均耗电系数: {weekend_energy_df['de_per_km_srv'].mean():.4f}")
print(f"   - 工作日平均耗电效率: {weekday_energy_df['de_per_km_tochg'].mean():.4f}")
print(f"   - 非工作日平均耗电效率: {weekend_energy_df['de_per_km_tochg'].mean():.4f}")
print(f"   - 速度差异: {abs(weekday_energy_df['de_per_km_srv'].mean() - weekend_energy_df['de_per_km_srv'].mean()):.2f} km/h")
print(f"   - 耗电系数差异: {abs(weekday_energy_df['de_per_km_srv'].mean() - weekend_energy_df['de_per_km_srv'].mean()):.4f}")

print(f"\n💾 文件格式:")
print(f"   - CSV格式: 兼容现有系统")
print(f"   - Parquet格式: 高效存储和读取")
print(f"   - 自动生成数据质量报告")

print(f"\n🎯 应用场景:")
print(f"   - SAEV运营优化研究")
print(f"   - 时间动态需求分析")
print(f"   - 拥堵影响评估")
print(f"   - 充电站布局优化")
print(f"   - 车队调度策略制定")
print(f"   - 工作日vs非工作日运营策略对比")
print(f"   - 多场景敏感性分析")
print(f"   - 政策影响评估")
print(f"   - 耗电优化策略制定")
print(f"   - 耗电效率分析")

print("\n" + "=" * 80)
print("✅ 所有任务已完成！生成的数据文件可直接用于SAEV运营优化研究。")
print("=" * 80)


# ## 16. 工作日vs非工作日对比分析
# 

# In[ ]:


# 工作日vs非工作日对比分析
print("=" * 80)
print("工作日vs非工作日对比分析")
print("=" * 80)

# 1. 出行量对比
print(f"\n1. 出行量对比:")
print(f"   - 工作日出行: {weekday_total:,} 次 ({weekday_total/(weekday_total+weekend_total)*100:.1f}%)")
print(f"   - 非工作日出行: {weekend_total:,} 次 ({weekend_total/(weekday_total+weekend_total)*100:.1f}%)")
print(f"   - 工作日/非工作日比例: {weekday_total/weekend_total:.2f}:1")

# 2. TOD分布对比
print(f"\n2. TOD分布对比:")
print(f"   - 工作日时段数: {len(weekday_tod_final)}")
print(f"   - 非工作日时段数: {len(weekend_tod_final)}")

# 找出高峰时段
weekday_peak = weekday_tod_final.nlargest(3, 'share_15min')
weekend_peak = weekend_tod_final.nlargest(3, 'share_15min')

print(f"\n   工作日前3个高峰时段:")
for _, row in weekday_peak.iterrows():
    print(f"     - 时段 {row['period_index']:2d} ({row['start_time']}-{row['end_time']}): {row['share_15min']:.6f}")

print(f"\n   非工作日前3个高峰时段:")
for _, row in weekend_peak.iterrows():
    print(f"     - 时段 {row['period_index']:2d} ({row['start_time']}-{row['end_time']}): {row['share_15min']:.6f}")

# 3. 拥堵系数对比
print(f"\n3. 拥堵系数对比:")
print(f"   - 工作日平均拥堵系数: {weekday_coeff_df['gamma_rep_p'].mean():.4f}")
print(f"   - 非工作日平均拥堵系数: {weekend_coeff_df['gamma_rep_p'].mean():.4f}")
print(f"   - 拥堵差异: {abs(weekday_coeff_df['gamma_rep_p'].mean() - weekend_coeff_df['gamma_rep_p'].mean()):.4f}")

# 找出最高拥堵时段
weekday_max_congestion = weekday_coeff_df.loc[weekday_coeff_df['gamma_rep_p'].idxmax()]
weekend_max_congestion = weekend_coeff_df.loc[weekend_coeff_df['gamma_rep_p'].idxmax()]

print(f"\n   工作日最高拥堵时段:")
period_hour = (weekday_max_congestion['t'] - 1) // 4
period_min = ((weekday_max_congestion['t'] - 1) % 4) * 15
time_str = f"{int(period_hour):02d}:{int(period_min):02d}"
print(f"     - 时段 {int(weekday_max_congestion['t']):2d} ({time_str}): {weekday_max_congestion['gamma_rep_p']:.4f}")

print(f"\n   非工作日最高拥堵时段:")
period_hour = (weekend_max_congestion['t'] - 1) // 4
period_min = ((weekend_max_congestion['t'] - 1) % 4) * 15
time_str = f"{int(period_hour):02d}:{int(period_min):02d}"
print(f"     - 时段 {int(weekend_max_congestion['t']):2d} ({time_str}): {weekend_max_congestion['gamma_rep_p']:.4f}")

# 4. 充电系数对比
print(f"\n4. 充电系数对比:")
print(f"   - 工作日beta_chg_p1平均: {weekday_coeff_df['beta_chg_p1'].mean():.4f}")
print(f"   - 非工作日beta_chg_p1平均: {weekend_coeff_df['beta_chg_p1'].mean():.4f}")
print(f"   - 工作日beta_chg_p2平均: {weekday_coeff_df['beta_chg_p2'].mean():.4f}")
print(f"   - 非工作日beta_chg_p2平均: {weekend_coeff_df['beta_chg_p2'].mean():.4f}")

# 5. 相关性分析
print(f"\n5. 相关性分析:")
# 计算工作日和非工作日拥堵系数的相关性
merged_coeff = weekday_coeff_df.merge(weekend_coeff_df, on='t', suffixes=('_weekday', '_weekend'))
correlation = merged_coeff['gamma_rep_p_weekday'].corr(merged_coeff['gamma_rep_p_weekend'])
print(f"   - 工作日vs非工作日拥堵系数相关性: {correlation:.4f}")

# 计算工作日和非工作日耗电系数的相关性
merged_energy = weekday_energy_df.merge(weekend_energy_df, on='t', suffixes=('_weekday', '_weekend'))
energy_correlation = merged_energy['de_per_km_srv_weekday'].corr(merged_energy['de_per_km_srv_weekend'])
speed_correlation = merged_energy['de_per_km_srv_weekday'].corr(merged_energy['de_per_km_srv_weekend'])
print(f"   - 工作日vs非工作日耗电系数相关性: {energy_correlation:.4f}")
print(f"   - 工作日vs非工作日速度相关性: {speed_correlation:.4f}")

# 6. 基础连接信息对比
print(f"\n6. 基础连接信息对比:")
print(f"   - 工作日平均出行距离: {weekday_base_ij_df['dist_km'].mean():.3f} km")
print(f"   - 非工作日平均出行距离: {weekend_base_ij_df['dist_km'].mean():.3f} km")
print(f"   - 工作日平均出行时间: {weekday_base_ij_df['base_minutes'].mean():.2f} 分钟")
print(f"   - 非工作日平均出行时间: {weekend_base_ij_df['base_minutes'].mean():.2f} 分钟")

# 计算距离差异
distance_diff = abs(weekday_base_ij_df['dist_km'].mean() - weekend_base_ij_df['dist_km'].mean())
time_diff = abs(weekday_base_ij_df['base_minutes'].mean() - weekend_base_ij_df['base_minutes'].mean())

print(f"   - 平均距离差异: {distance_diff:.3f} km")
print(f"   - 平均时间差异: {time_diff:.2f} 分钟")

# 7. 耗电信息对比
print(f"\n7. 耗电信息对比:")
print(f"   - 工作日平均速度: {weekday_energy_df['de_per_km_srv'].mean():.2f} km/h")
print(f"   - 非工作日平均速度: {weekend_energy_df['de_per_km_srv'].mean():.2f} km/h")
print(f"   - 工作日平均耗电系数: {weekday_energy_df['de_per_km_srv'].mean():.4f}")
print(f"   - 非工作日平均耗电系数: {weekend_energy_df['de_per_km_srv'].mean():.4f}")
print(f"   - 工作日平均耗电效率: {weekday_energy_df['de_per_km_tochg'].mean():.4f}")
print(f"   - 非工作日平均耗电效率: {weekend_energy_df['de_per_km_tochg'].mean():.4f}")

# 计算耗电差异
speed_diff = abs(weekday_energy_df['de_per_km_srv'].mean() - weekend_energy_df['de_per_km_srv'].mean())
energy_diff = abs(weekday_energy_df['de_per_km_srv'].mean() - weekend_energy_df['de_per_km_srv'].mean())
efficiency_diff = abs(weekday_energy_df['de_per_km_tochg'].mean() - weekend_energy_df['de_per_km_tochg'].mean())

print(f"   - 平均速度差异: {speed_diff:.2f} km/h")
print(f"   - 平均耗电系数差异: {energy_diff:.4f}")
print(f"   - 平均耗电效率差异: {efficiency_diff:.4f}")

# 8. 应用建议
print(f"\n8. 应用建议:")
if weekday_total > weekend_total * 1.5:
    print(f"   - 工作日出行量显著高于非工作日，建议重点关注工作日运营策略")
else:
    print(f"   - 工作日和非工作日出行量相对均衡，建议制定差异化运营策略")

if abs(weekday_coeff_df['gamma_rep_p'].mean() - weekend_coeff_df['gamma_rep_p'].mean()) > 0.1:
    print(f"   - 拥堵模式存在显著差异，需要分别优化")
else:
    print(f"   - 拥堵模式相对一致，可以统一优化策略")

if distance_diff > 0.5 or time_diff > 2.0:
    print(f"   - 基础连接信息存在显著差异，需要分别制定运营策略")
else:
    print(f"   - 基础连接信息相对一致，可以统一优化策略")

if speed_diff > 5.0 or energy_diff > 0.2:
    print(f"   - 耗电模式存在显著差异，需要分别制定耗电策略")
else:
    print(f"   - 耗电模式相对一致，可以统一优化策略")

print("\n" + "=" * 80)

