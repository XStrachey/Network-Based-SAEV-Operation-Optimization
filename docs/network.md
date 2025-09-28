# 网络化SAEV运营优化模型（Min-Cost Flow over a time–SOC network）

## 模型概览

我们在离散化的时空—电量网格上构建最小费用流优化模型，用于优化共享自动驾驶电动汽车(SAEV)的运营调度。

### 核心组件

**节点类型**：
- **真实网格节点** $(i,t,\ell)$：区域 $i$、时间步 $t$、SOC 等级 $\ell$
- **服务闸门节点** $\mathrm{svc\_in}(i,j,t),\mathrm{svc\_out}(i,j,t)$：服务需求约束
- **充电队列节点** $\mathrm{q\_in}(k,p),\mathrm{q\_out}(k,p)$：充电站并发容量约束
- **总车队源节点** $\mathrm{source\_total}$：代表总车队规模，为初始节点提供入度
- **超级汇点** $\mathrm{sink}$（可选）：网络流平衡

**弧类型**：按功能分为 4 大类 10 种细分类型，每种弧都有特定的容量约束和成本结构。

### 数学建模

**目标函数**：
$$\min \sum_{a \in \mathcal{A}} c_a x_a$$

其中：
- $c_a$ 是弧 $a$ 的单位成本（包含时间成本、占位成本、奖励等）
- $x_a$ 是弧 $a$ 上的流量（连续变量）

**约束条件**：
1. **流守恒约束**：$\sum_{a \in \delta^-(v)} x_a - \sum_{a \in \delta^+(v)} x_a = s_v, \quad \forall v \in \mathcal{V}$
2. **容量约束**：$0 \leq x_a \leq u_a, \quad \forall a \in \mathcal{A}$
3. **非负约束**：$x_a \geq 0, \quad \forall a \in \mathcal{A}$

其中：
- $\delta^-(v)$ 和 $\delta^+(v)$ 分别是以节点 $v$ 为入端和出端的弧集合
- $s_v$ 是节点 $v$ 的供给量（正值为源，负值为汇）
- $u_a$ 是弧 $a$ 的容量上限

**流守恒约束的完整性**：
- **总车队源节点**：$s_{source} = +\sum_{i,\ell} V_0(i,\ell)$，为所有初始节点提供入度
- **中间节点**：$s_v = 0$，入流等于出流
- **超级汇点**：$s_{sink} = -\sum_{i,\ell} V_0(i,\ell)$，吸收所有最终流量
- 这确保了网络中每个节点都有完整的入度和出度（除了源点和汇点）

## 新增功能特性

### 可视化分析系统
- **SAEVResultVisualizer**: 完整的求解结果可视化工具
- **多维度分析**: 流量分布、成本效益、时空演化、网络拓扑
- **交互式图表**: 支持plotly、folium等交互式可视化
- **自动化报告**: 生成完整的分析报告
- **实时监控**: 支持滚动窗口求解的实时结果展示
- **地图集成**: 基于地理信息的流量可视化

### 管道执行系统
- **run_pipeline.py**: 支持顺序执行多个脚本步骤
- **run_rolling_solve.py**: 滚动窗口求解的简化接口
- **日志记录**: 完整的执行日志和错误处理
- **超时控制**: 可配置的执行超时机制
- **重试机制**: 支持失败重试
- **实时输出**: 支持求解过程的实时状态显示
- **回调函数**: 支持窗口完成后的自定义处理

### 增强的求解器系统
- **负环检测**: 求解前自动检测负成本环，避免无界解
- **多求解器支持**: GLPK、CBC等求解器自动选择
- **性能优化**: 大规模问题的求解优化
- **滚动窗口**: 支持大规模时间范围的滚动优化
- **内存管理**: 优化的内存使用和垃圾回收
- **并行处理**: 支持多窗口并行求解
- **数值稳定性**: 自动处理无穷大容量值，确保PuLP求解器兼容性

### 配置管理系统
- **NetworkConfig**: 统一的配置管理类
- **模块化配置**: 支持分模块的配置管理
- **默认值**: 所有配置参数都有合理的默认值
- **环境变量**: 支持环境变量覆盖配置
- **配置文件**: JSON格式的配置文件支持

### R1重定位弧定向裁剪系统
- **需求压力梯度导向**: 基于供需不平衡的智能弧裁剪
- **数学建模**: 完整的需求压力DP(i,t)计算和梯度分析
- **定向规则**: 智能保留"朝压力更高"方向的重定位弧
- **安全兜底**: 最小度约束、连通性检查、反向边保留
- **性能优化**: 预期减少35%+重定位弧，显著提升求解速度
- **可回滚设计**: 支持--no-r1-prune参数完全禁用

---

# 一、集合、索引与输入

**时间与电量**

* 基础步长：$\Delta t=$ `cfg.time_soc.dt_minutes`（分钟，默认15分钟）。
* 窗口：给定 $t_0$ 与 $H$，令 $t_{\mathrm{hi}}=\min({t_{end\_step}},\, t_0+H)$ 。
* Halo：到达上限 $t_{\max}^{\text{arr}}=\min(t_{\mathrm{hi}}+B,\ \max(\mathcal T))$，其中 $B=$ `overhang_steps`，默认6步）。
* SOC 等级集合 $\mathcal L=\{0,5,\dots,100\}$（或配置等距集合），记间距 $\Delta_\ell$。
* 滚动步长：`cfg.time_soc.roll_step`（默认1步）。

**区域与站点**

* 区域集合 $\mathcal Z$ 来自 `zones.csv`。
* 站点集合 $\mathcal K$，每站点 $k$ 有**所在区域** $\mathrm{zone}(k)$ 与**桩功率等级** $\mathrm{level}(k)$（来自 `stations.csv` / 映射函数）。
* 最近站集合：对每个区域 $i$，取按 `dist_km` 升序的前 $n$ 个站（`cfg.pruning.charge_nearest_station_n`，默认 8），记为 $\mathcal K_i^{\text{near}}$。
* 站点容量：每站点 $k$ 有**插座数** $\mathrm{plugs}_k$ 和**利用率因子** $\mathrm{util\_factor}_k$。

**OD 与“服务闸门”需求**

* 仅对 `od_matrix` 中 $(i,j,t)$ 需求 $D_{ijt}>0$ 生成服务闸门弧。

**行驶时间与能耗（由基础表与能耗系数驱动）**

* OD 行驶时间：$\tau_{ij}=\left\lceil \frac{\texttt{dist\_km}_{ij}}{\texttt{avg\_speed\_kmh}/60}\frac{1}{\Delta t}\right\rceil$。
* 区域 $\to$ 站点时间：$\tau_{i\to k}=\left\lceil \frac{\texttt{dist\_km}_{ik}}{\texttt{avg\_speed\_kmh}/60}\frac{1}{\Delta t}\right\rceil$，其中 `avg_speed_kmh` 来自 `cfg.basic.avg_speed_kmh`（默认30 km/h）。
* 多步能耗离散：对弧类型 $r\in\{\text{srv},\text{rep},\text{tochg}\}$，在 $[t,t+\tau-1]$ 区间按 `coeff_energy` 的每步"每公里耗电" $e^{(r)}_{t'}$ 汇总并映射到 SOC 等级整数：

  $$
    \Delta\ell^{(r)}(t,\tau,d) \;=\; \text{RoundToSOC}\!\left(\sum_{q=0}^{\tau-1} e^{(r)}_{t+q}\cdot d\right)\in \mathcal L
  $$

  （与实现中 `compute_multi_timestep_energy_consumption` 等价；结果取非负整数 SOC 点。）
* 能耗系数：来自 `coeff_energy.csv` 或 `cfg.energy` 配置，包括 `de_per_km_srv`、`de_per_km_rep`、`de_per_km_tochg`。

**Reachability（03，ALL-NEAREST 策略）**

* 先为每个 $(j,t)$ 和 $k\in\mathcal K_j^{\text{near}}$ 计算到站能耗 $\Delta\ell^{(\text{tochg})}_{jkt}$。
* 定义可达节点集合

  $$
  \mathcal R=\bigl\{(j,t,\ell)\,:\, \ell \ge \max_{k\in\mathcal K_j^{\text{near}}}\Delta\ell^{(\text{tochg})}_{jkt}\ \text{且所有该 \(\Delta\ell\) 可计算}\bigr\}.
  $$
* **弧生成时**要求相关起/到节点均在 $\mathcal R$（详见弧定义）。

**充电曲线（分钟表）**

* 对每站功率等级 $L$ 给出 $\tau_{\text{chg}}(L;\ \ell_{\text{in}}\!\to\!\ell_{\text{out}})$（来自 `load_or_build_charging_profile`），换算步数 $\left\lceil \frac{\tau_{\text{chg}}}{\Delta t}\right\rceil$。
* 单次充电的**目标 SOC 增量**：$\delta_{\min}=$ `cfg.charge_queue.min_charge_step`（默认20），对起始 $\ell$ 令 $\ell^{\star}=\ell+\min\{\delta_{\min},\,100-\ell\}$。
* 充电站容量：$u_{k,p}=\max\{1,\lfloor \mathrm{plugs}_k \times \mathrm{util\_factor}_k \times \mathrm{queue\_relax\_factor} \rfloor\}$，其中 `queue_relax_factor` 来自 `cfg.charge_queue.queue_relax_factor`（默认1.2）。

**初始供给与网络平衡**

* 初始库存 $V_0(i,\ell)$（仅 $t=t_0$）来自 `initial_inventory.parquet`。
* **总车队源节点** $\mathrm{source\_total}$：供给量 $s_{source} = +\sum_{i,\ell} V_0(i,\ell)$，连接到所有有初始库存的节点。
* **超级汇点** $\mathrm{sink}$（可选）：供给量 $s_{sink} = -\sum_{i,\ell} V_0(i,\ell)$，所有 $t=t_{\mathrm{hi}}$ 的网格节点连到 $\mathrm{sink}$。
* 这种设计确保了完整的流守恒约束：每个中间节点都有入度和出度。

---

# 二、节点与弧（04 + 07）

**节点集合**

$$
\mathcal V=\underbrace{\{(i,t,\ell)\in\mathcal Z\times\mathcal T\times\mathcal L\}}_{\text{网格节点}}
\,\cup\,\{\mathrm{svc\_in}(i,j,t),\ \mathrm{svc\_out}(i,j,t)\}\,\cup\,\{\mathrm{q\_in}(k,p),\ \mathrm{q\_out}(k,p)\}\,\cup\,\{\mathrm{source\_total},\ \mathrm{sink}\ \text{(可选)}\}.
$$

**弧集合**（仅生成“**出发时刻** $t\in[t_0,t_{\mathrm{hi}}-1]$”的弧；允许“**到达** $\le t_{\max}^{\text{arr}}$”）

* **Idle**：$(i,t,\ell)\to(i,t\!+\!1,\ell)$，若 $t+1\le\max\mathcal T$。

* **Service（三段闸门）**（仅对 $D_{ijt}>0$）

  * 进入：$(i,t,\ell)\to \mathrm{svc\_in}(i,j,t)$，若 $\ell\ge \Delta\ell^{(\text{srv})}(t,\tau_{ij},d_{ij})$ 且 $(i,t,\ell),(j,t+\tau_{ij},\ell-\Delta\ell)\in\mathcal R$。
  * 闸门（容量弧）：$\mathrm{svc\_in}(i,j,t)\to \mathrm{svc\_out}(i,j,t)$，容量 $=D_{ijt}$。
  * 退出：$\mathrm{svc\_out}(i,j,t)\to (j,t+\tau_{ij},\ell-\Delta\ell)$。

* **Reposition**（i≠j，$\tau_{ij}\le\left\lceil\frac{\texttt{max\_reposition\_tt}}{\Delta t}\right\rceil$）

  $$
  (i,t,\ell)\to (j,t+\tau_{ij},\ell-\Delta\ell^{(\text{rep})}(t,\tau_{ij},d_{ij})),
  $$

  仅在 $\ell\ge\max\{\Delta\ell^{(\text{rep})},\ \texttt{cfg.pruning.min\_soc\_for\_reposition}\}$（默认20）且起/到节点 $\in\mathcal R$ 时生成。
  还将对**同一起点节点**的重定位弧做 KNN 剪枝：每个 `from_node_id` 仅保留按 `dist_km`（或 $\tau$ 兜底）最近的 $K=$ `cfg.pruning.reposition_nearest_zone_n`（默认8）条。

* **Charging**（四段占用链，容量来自站点并发）（仅 $k\in\mathcal K_i^{\text{near}}$）
  
  * 去站：$(i,t,\ell)\to (\mathrm{zone}(k),\ t+\tau_{i\to k},\ \ell-\Delta\ell^{(\text{tochg})}(t,\tau_{i\to k},d_{ik}))$，要求 $\ell$ 不小于该能耗，且相关节点 $\in\mathcal R$。
  * 进入占位：$(\mathrm{zone}(k),p,\ell_p)\to \mathrm{q\_in}(k,p)$。
  * 占位（容量弧）：$\mathrm{q\_in}(k,p)\to \mathrm{q\_out}(k,p)$，容量

  $$
  u_{k,p}\;=\;\hat c_k\;\;\equiv\;\max\!\Bigl\{1,\ \varphi\bigl(\mathrm{plugs}_k\cdot \mathrm{util\_factor}_k \cdot \mathrm{queue\_relax\_factor}\bigr)\Bigr\},
  $$

  其中 $\mathrm{plugs}_k,\ \mathrm{util\_factor}_k$ 来自 `stations.csv`，$\mathrm{queue\_relax\_factor}$ 来自 `cfg.charge_queue.queue_relax_factor`（默认1.2），$\varphi(\cdot)$ 缺省取 **$\lfloor\cdot\rfloor$**（保守口径，不超过物理并发；如需"软容量"可改为 $\lceil\cdot\rceil$）。若站点数据缺失，回退到 `cfg.charge_queue.default_plugs_per_station`（默认1）。注意本实现中 $\hat c_k$ **对同一站点随 $p$ 不变**（即 $u_{k,p}\equiv \hat c_k$）。
  
  * 逐步推进：$\mathrm{q\_out}(k,p)\to (\mathrm{zone}(k),p\!+\!1,\ell_{p+1})$。若 $p$ 为最后一步则 $\ell_{p+1}=\ell^\star$，否则 $\ell_{p+1}=\ell_p$。
  其中充电步数

  $$
  \tau_{\text{chg}}=\Bigl\lceil \tfrac{\tau_{\text{chg}}(\mathrm{level}(k);\ \ell_{\text{arr}}\!\to\!\ell^\star)}{\Delta t}\Bigr\rceil,
  $$

  并要求整个链条的每个网格节点都在 $\mathcal R$。

> 口径说明：$\mathrm{util\_factor}_k\in[0,1]$ 表示期望利用率（或可用率）；$\mathrm{queue\_relax\_factor}\ge 1$ 允许在网络流里**放松**并发上限（例如考虑排队“软容量”）。保守设置推荐 $\varphi=\lfloor\cdot\rfloor$。

* **From-source**：从总车队源节点 $\mathrm{source\_total}$ 到所有有初始库存的节点 $(i,t_0,\ell)$，容量等于初始库存 $V_0(i,\ell)$，成本为 $\texttt{cost\_from\_source}$（默认 0）。

* **To-sink（可选）**：对于所有 $t=t_{\mathrm{hi}}$ 的网格节点 $v$，添加 $v\to \mathrm{sink}$ 的弧，容量为大数 $U$（默认 $10^{12}$），成本为 $\texttt{cost\_to\_sink}$（默认 0）。

> 实现细节一致性：所有弧都会**剔除自环** $(\texttt{from}=\texttt{to})$；服务闸门本身构造即可避免自环，成本模块（06）仍对异常自环做零化保护。

---

# 三、决策变量

对每条生成的弧 $a\in\mathcal A$ 引入流量 $x_a\ge 0$。模型为**线性连续流**（01 中 `RELAX_BINARIES=True`）。

---

# 四、容量与供给

**弧容量**

$$
0\ \le\ x_a\ \le\ u_a\quad \forall a\in\mathcal A,
$$

其中

* 服务闸门弧：$u_a=D_{ijt}$；
* 充电占位弧：充电占位弧 $u_{k,p}=\hat c_k=\max\{1,\varphi(\mathrm{plugs}_k\cdot \mathrm{util\_factor}_k\cdot \mathrm{queue\_relax\_factor})\}$；
* 其他弧：$u_a=U$（大数，上层实现用 $10^{12}$），仅由供给与连通性间接限流。

**节点供给**
令节点供给 $s_v$：

* $s_{\mathrm{source\_total}}=+\sum_{i,\ell}V_0(i,\ell)$（总车队源节点）；
* $s_{(i,t_0,\ell)}=0$（初始库存节点，通过from-source弧接收流量）；
* 若使用超级汇点：$s_{\mathrm{sink}}=-\sum_{i,\ell}V_0(i,\ell)$；
* 其余节点 $s_v=0$。

---

# 五、流守恒约束

$$
\sum_{a\in\delta^-(v)} x_a\ -\ \sum_{a\in\delta^+(v)} x_a \;=\; s_v,\qquad \forall v\in\mathcal V.
$$

> 说明：$\delta^{-}(v)$ 与 $\delta^{+}(v)$ 分别是以 $v$ 为入端/出端的弧集。
> 滚动窗口允许弧“跨窗”到达（Halo），因此**不强制**窗口末端可达（除非在 05/07 里显式启用 `require_bwd=True`）。

---

# 六、成本结构（时间可变系数 + 奖励机制）

## 总成本函数

每条弧的总成本由以下组件构成：

$$c_a = c^{\text{rep}}_a + c^{\text{chg-travel}}_a + c^{\text{occ}}_a + c^{\text{svc-gate}}_a + c^{\text{rep-reward}}_a + c^{\text{chg-reward}}_a + c^{\text{idle}}_a$$

其中未涉及的类型其对应项取 0。

## 成本组件详细定义

### 1. 重定位成本 $c^{\text{rep}}_a$

**适用弧类型**：`reposition`

**数学表达式**：
$$c^{\text{rep}}_a = \mathrm{VOT} \cdot \sum_{q=0}^{\tau-1} \gamma_{\text{rep\_p}}(t+q)$$

**实现细节**：
- $\mathrm{VOT}$：时间价值系数（默认1.0）
- $\gamma_{\text{rep\_p}}(t)$：时间可变的重定位成本系数
- 来源：`coeff_schedule.csv` 或配置常数 `cfg.costs_equity.gamma_rep`

### 2. 去充电行驶成本 $c^{\text{chg-travel}}_a$

**适用弧类型**：`tochg`

**数学表达式**：
$$c^{\text{chg-travel}}_a = \mathrm{VOT} \cdot \sum_{q=0}^{\tau_{\text{tochg}}-1} \beta_{\text{chg\_p1}}(t+q)$$

**实现细节**：
- $\beta_{\text{chg\_p1}}(t)$：时间可变的去充电行驶成本系数
- 来源：`coeff_schedule.csv` 或配置常数 `cfg.costs_equity.beta_toCHG`

### 3. 充电占位成本 $c^{\text{occ}}_a$

**适用弧类型**：`chg_occ`

**数学表达式**：
$$c^{\text{occ}}_a = \mathrm{VOT} \cdot \beta_{\text{chg\_p2}}(p)$$

**实现细节**：
- $\beta_{\text{chg\_p2}}(p)$：充电占位成本系数
- $p$ 是该占位步的时刻
- 来源：`coeff_schedule.csv` 或配置常数 `cfg.costs_equity.beta_chg`

### 4. 服务奖励 $c^{\text{svc-gate}}_a$（负成本）

**适用弧类型**：`svc_gate`

**数学表达式**：
$$c^{\text{svc-gate}}_a = -\mathrm{VOT} \cdot w_{ijt}$$

**实现细节**：
- $w_{ijt}$：服务权重，默认来自 `cfg.costs_equity.unmet_weight_default`
- 可被 `cfg.unmet_weights_overrides[t][(i,j)]` 覆盖
- 受 `cfg.flags.enable_service_reward` 控制

### 5. 重定位收益 $c^{\text{rep-reward}}_a$（负成本）

**适用弧类型**：`reposition`

**数学表达式**：
$$c^{\text{rep-reward}}_a = -\gamma_{\text{rep}} \cdot \text{zone\_value}(t,j)$$

**区域价值计算**：
$$\text{zone\_value}(t,j) = \max\left\{\sum_i D_{ijt}^{\text{out}} - \text{inv}_0(j), 0\right\}$$

**实现细节**：
- $\gamma_{\text{rep}}$：重定位收益系数（默认0.2）
- 归一化模式：`per_t_sum`, `per_t_max`, `global_max`, `window_sum`, `none`
- 受 `cfg.flags.enable_reposition_reward` 控制

### 6. 充电收益 $c^{\text{chg-reward}}_a$（负成本）

**适用弧类型**：`chg_occ`, `chg_step`

**数学表达式**：
$$c^{\text{chg-reward}}_a = -\alpha_{\text{chg}} \cdot \max\{\ell_{\text{to}} - \ell, 0\}$$

**实现细节**：
- $\alpha_{\text{chg}}$：充电收益系数（默认0.02）
- 基于SOC变化量计算
- 受 `cfg.flags.enable_charging_reward` 控制

### 7. Idle机会成本 $c^{\text{idle}}_a$

**适用弧类型**：`idle`

**数学表达式**：
$$c^{\text{idle}}_a = \text{idle\_opportunity\_cost}$$

**实现细节**：
- 常数成本（默认10.0）
- 来源：`cfg.costs_equity.idle_opportunity_cost`

**时间可变系数（$\gamma_{rep_p},\beta_{chg_{p1}},\beta_{chg_{p2}}$，带后备常数）**

* 来自 `coeff_schedule.csv` 的时间表（列 $(t,\gamma_{rep_p},\beta_{chg_{p1}},\beta_{chg_{p2}})$），若缺失则用 `cfg.costs_equity` 中的常数后备。
* 统一乘以 $\mathrm{VOT}$（`cfg.costs_equity.vot`，默认1.0）。
* 后备常数：`gamma_rep`（默认1.0）、`beta_toCHG`（默认1.0）、`beta_chg`（默认1.0）。

**（1）重定位成本**（仅 `reposition` 弧）

$$
c^{\text{rep}}_a \;=\; \mathrm{VOT}\cdot \sum_{q=0}^{\tau_{ij}-1}\gamma_{rep_p}(t+q).
$$

**（2）去充电行驶成本**（仅 `tochg` 弧）

$$
c^{\text{chg-travel}}_a \;=\; \mathrm{VOT}\cdot \sum_{q=0}^{\tau_{i\to k}-1}\beta_{chg_{p1}}(t+q).
$$

**（3）充电占位成本**（仅 `chg_occ` 弧）

$$
c^{\text{occ}}_a \;=\; \mathrm{VOT}\cdot \beta_{chg_{p2}}(p),
$$

其中 $p$ 是该占位步的时刻。

**（4）服务奖励（负成本，仅 `svc_gate` 弧）**

$$
c^{\text{svc-gate}}_a \;=\; -\,\mathrm{VOT}\cdot w_{ijt},
$$

默认 $w_{ijt}=$ `cfg.costs_equity.unmet_weight_default`（默认1.0），可被 `cfg.unmet_weights_overrides[t][(i,j)]` 覆盖。受 `cfg.flags.enable_service_reward` 控制。

> 作用等价于传统“未满足惩罚”的对偶形式：闸门容量 $D_{ijt}$ 决定最多可服务量，负成本鼓励通过闸门以满足需求。

**（5）重定位收益（负成本，仅 `reposition` 弧）**
定义“区域价值”

$$
\mathrm{zone\_value}(t,j)=\max\Bigl\{\sum_{i} D_{ijt}^{\text{out}} - \mathrm{inv}_0(j),\; 0\Bigr\},
$$

其中 $D_{ijt}^{\text{out}}$ 是**以 $j$ 为起点**的当期总对外需求，$\mathrm{inv}_0(j)$ 是 $t_0$ 的期初库存（按区汇总）。
收益（负成本）系数 $\gamma_{\text{rep}}=$ `cfg.costs_equity.gamma_reposition_reward`（默认1.0）：

$$
c^{\text{rep-reward}}_a \;=\; -\,\gamma_{\text{rep}}\cdot \mathrm{zone\_value}(t,j).
$$

受 `cfg.flags.enable_reposition_reward` 控制。

**（6）充电收益（负成本，理论上用于 `chg_occ`/`chg_step`）**
期望形式为

$$
c^{\text{chg-reward}}_a \;=\; -\,\beta_{\text{chg}}\cdot \max\{\ell_{\text{to}}-\ell,\,0\}.
$$

受 `cfg.flags.enable_charging_reward` 控制。

> **实现注记（与代码一致）：** 由于 04 中 `chg_occ`/`chg_step` 实际列集未包含 $\ell_{\text{to}}$，06 的该项**合并不到**弧表，于是该项在当前脚本组合下为 **0**（即“充电收益”未生效）。若上游提供 `l_to` 列，则自动启用。

**Idle/enter/exit/step 等其他弧**：费用为 0（除上面适用的项）。

---

# 七、连通性裁剪与窗口策略（05 + 07）

* **生成策略**：仅生成“出发时刻落在窗口 $[t_0,t_{\mathrm{hi}}-1]$”的弧；允许到达跨窗（Halo）。
* **Reachability 端点过滤**：除 Idle 外，弧生成阶段会要求起/到节点在 $\mathcal R$（详见二）。
* **BFS 连通性裁剪**（05）：

  * 源集合 $S$：由 $t=t_0$ 且 $V_0(i,\ell)>0$ 的网格节点求出；
  * 终端集合 $F$：$t=t_{\mathrm{hi}}$ 的网格节点（仅在 `require_bwd=True` 时参与 BWD）。
  * 记 $\mathrm{FWD}=\mathrm{BFS}_{\to}(S)$，$\mathrm{BWD}=\mathrm{BFS}_{\leftarrow}(F)$。
  * 保留集合 $\mathrm{KEEP}=\mathrm{FWD}$（默认）或 $\mathrm{FWD}\cap\mathrm{BWD}$（若要求 BWD）。
  * **保留规则** `keep_on`：

    * `'from'`（默认）：仅保留 $\texttt{from\_node}\in\mathrm{KEEP}$ 的弧；
    * `'both'`：两端都在 $\mathrm{KEEP}$ 才保留；
    * `'either'`：任一端在 $\mathrm{KEEP}$ 即保留。
* **自环剔除**：在各类弧生成后再次兜底删除 $\texttt{from}=\texttt{to}$。

---

# 七点五、需求驱动的重定位弧生成

## 背景与动机

在SAEV运营中，重定位弧数量往往占整个网络的很大比例（通常可达数十万条），直接影响求解效率。传统的生成所有可能OD对的方法会产生大量不必要的弧。

**需求驱动的重定位弧生成**通过在生成端就控制弧的数量，只对有实际业务需求的重定位路径生成弧，从而在源头解决弧数量过多的问题。

## 核心思想

1. **不引入虚节点**：保持直接弧设计，避免增加图结构复杂度
2. **需求驱动生成**：只对有重定位需求的OD对生成弧
3. **智能过滤**：基于供需不平衡、距离、需求阈值等多维度筛选

## 需求计算逻辑

### 方法1：基于高服务需求的OD对

选择服务需求前20%的OD对，按比例生成重定位需求：

$$\text{重定位需求}_{ijt} = \text{服务需求}_{ijt} \times \text{reposition\_demand\_ratio}$$

### 方法2：基于供需不平衡的逆向重定位

计算每个区域每个时刻的供需不平衡：

$$\text{不平衡}_{it} = D_i(t) - S_i(t)$$

从供给过剩的区域到需求过剩的区域生成重定位需求。

### 过滤条件

1. **需求阈值过滤**：过滤掉需求小于 `min_reposition_demand` 的OD对
2. **时间约束**：保持原有的最大重定位时间 (`max_reposition_tt`) 和SOC约束

## 配置参数

```python
@dataclass
class PruningRules:
    # 需求驱动的重定位弧生成参数
    reposition_demand_ratio: float = 0.3          # 重定位需求相对于服务需求的比例
    min_reposition_demand: float = 0.1            # 最小重定位需求阈值
    reposition_imbalance_threshold: float = 1.0   # 供需不平衡阈值
    # 注意：距离约束通过 max_reposition_tt 统一控制
```

## 预期效果

- **弧数量减少**：预期减少50%以上的重定位弧
- **求解效率提升**：减少LP/MIP问题规模
- **保持解质量**：只生成有业务意义的重定位选项
- **无额外复杂度**：不引入虚节点，保持图结构简洁

## 集成方式

需求驱动的重定位弧生成集成在**弧生成阶段**（04）：

1. **弧生成**（04）：需求驱动的重定位弧生成 ⭐
2. **连通性裁剪**（05）：基础连通性过滤
3. **成本附加**（06）：计算弧成本
4. **容量设置**：设置弧容量
5. **求解器准备**：最终图结构

系统现在完全采用需求驱动的重定位弧生成方法，不再支持传统方法。

---

# 八、完整优化问题

## 标准最小费用流问题

令 $\mathcal A$ 为裁剪后的弧集，$\mathcal V$ 为实际被弧触达的节点集（含伪节点与可选汇点），则最终优化问题为：

$$
\begin{aligned}
\min_{x}&\quad \sum_{a\in\mathcal A} c_a\,x_a \\
\text{s.t.}&\quad \sum_{a\in\delta^{-}(v)} x_a - \sum_{a\in\delta^{+}(v)} x_a = s_v, && \forall v\in\mathcal V,\\
&\quad 0 \le x_a \le u_a, && \forall a\in\mathcal A,\\
&\quad x_a \in \mathbb{R}_{\ge 0}. &&
\end{aligned}
$$

## 求解器实现

### 1. 线性规划求解器（GLPK/CBC）

**实现文件**：`_02_solve_graph_mincost.py`

**特点**：
- 使用 PuLP 库构建线性规划模型
- 支持 GLPK 和 CBC 求解器自动选择
- 支持超时控制和求解状态监控
- 输出最优流量分配和求解统计

**求解流程**：
1. 读取图结构（节点和弧）
2. 数据清洗和类型转换
3. 供需平衡检查
4. 构建线性规划模型
5. 选择求解器并求解
6. 输出结果和统计信息

### 2. 网络单纯形求解器（SSP）

**实现文件**：`_03_network_simplex_solver.py`

**特点**：
- 使用 Successive Shortest Path (SSP) 算法
- 带势维护的约化费用计算
- 支持大规模网络的高效求解
- 内存优化的残量图表示

**算法流程**：
1. 构建残量图
2. 初始化势向量
3. 主循环：寻找最短路径并增广
4. 更新势向量
5. 直到所有供给分配完毕

### 3. 求解器选择策略

**自动选择**：
- 优先使用 GLPK（开源，稳定）
- 失败时回退到 CBC（功能更全面）
- 支持超时控制和错误处理

**性能对比**：
- 小规模问题：GLPK 更快
- 大规模问题：CBC 更稳定
- 网络单纯形：内存效率更高

---

# 九、与实现逐点对齐的要点/细节

1. **只为有需求的 $(i,j,t)$** 生成服务弧，并用**单条** `svc_gate` 容量弧承载 $D_{ijt}$ 与**负成本奖励**；`svc_enter/exit` 仅作连接。
2. **充电**用"占用链"：每一步都要穿过 `chg_occ`（容量=并行桩数），SOC 在**最后一步**一次性从 $\ell_{\text{arr}}$ 跳到 $\ell^\star$。
3. **Reachability（ALL-NEAREST）**采用"**必须能到全部最近站**"的判据：$\ell$ 至少覆盖最近站集合里**最大**的到站能耗。
4. **跨窗允许**：生成弧只看**出发时刻**是否在窗内，到达可延伸到 $t_{\mathrm{hi}}+B$。
5. **连通性裁剪**默认仅做 FWD，且 `'from'` 侧保留，确保"能从 $t_0$ 的供给出发"的弧都保留，即使到达在窗外；若启用 `require_bwd` 则还需能走到窗末节点。
6. **KNN 剪枝**仅作用于 `reposition` 的**出度**，度量优先 `dist_km`，兜底 `tau`。
7. **R1重定位弧定向裁剪**基于需求压力梯度，智能删除"逆梯度"的重定位弧：
   - 计算需求压力 $\text{DP}(i,t) = D_i(t) - S_i(t)$
   - 对对称重定位邻居 $(i \leftrightarrow j)$，计算压力梯度差 $\Delta\text{DP} = \text{DP}(j,t+\tau) - \text{DP}(i,t)$
   - 保留朝压力更高方向的弧，删除逆梯度弧，在不确定带内保留双向
   - 应用最小度约束和连通性检查确保安全性
   - 集成在成本附加之后、容量设置之前，支持 `--no-r1-prune` 禁用
8. **费用系数**来自 `coeff_schedule`（缺省用常数），并统一乘以 VOT；

   * `svc_gate` 负成本 = VOT × unmet\_weight（可 override）；
   * `reposition` 存在**额外负成本** = $-\gamma_{\text{rep}}\cdot\mathrm{zone\_value}$；
   * **当前脚本组合下"充电收益"实为 0**（因缺少 $\ell_{\text{to}}$ 列，06 的奖励合并不到 `chg_*` 弧）。
9. **容量设置**：只有 `svc_gate` 与 `chg_occ` 使用有限容量（分别为需求与并行桩数），其它弧容量为大数。
10. **Idle 弧**费用恒 0；服务/充电的 `enter/exit/step` 弧也无费用。
11. **超级汇点**（可选）把 $t=t_{\mathrm{hi}}$ 的所有网格节点连接到 `sink`，并在节点供给向量里加上等额负供给以**闭合**网络。
12. **自环**在建弧阶段与落盘前都会剔除；06 再把异常出现的 `svc_gate` 自环费用置 0，避免"空刷奖励"。
13. **配置系统**：使用 `NetworkConfig` 类统一管理所有配置参数，支持模块化配置。
14. **求解器增强**：支持负环检测、多求解器选择、超时控制等高级功能。
15. **可视化系统**：提供完整的求解结果可视化工具，支持多维度分析。
16. **滚动窗口优化**：支持大规模时间范围的滚动优化，提高求解效率。
17. **实时监控**：支持求解过程的实时状态显示和结果回调。
18. **内存优化**：针对大规模问题的内存使用优化和垃圾回收机制。
19. **并行处理**：支持多窗口并行求解，提高整体性能。

# 十、虚节点

## 为什么要引入虚节点？

* 目的是把“**容量/需求**”与“**时空推进/能耗**”解耦。
* 具体做法：把一条原本“从 A 到 B 的复合动作”切成多段：

  * 中间段是**唯一的容量弧**（承载“数量上限/奖励/惩罚”）；
  * 两侧段是**连接弧**（把车辆从网格节点送到容量弧，再从容量弧送回网格，负责时间推进与 SOC 变化）。
* 这样既能“一条弧表达容量”，又能保留“不同 SOC、不同起点”的细粒度可行性。

---

## （一）、服务弧的虚节点与转移（svc\_in / svc\_out）

### 节点与弧

对每个有需求的 $(i,j,t)$（`od_matrix` 中 demand>0），生成**两个虚节点**：

* $\mathrm{svc\_in}(i,j,t)$
* $\mathrm{svc\_out}(i,j,t)$

以及三类弧（对应 04 的 `svc_enter / svc_gate / svc_exit`）：

1. **进入弧** `svc_enter`（$\tau=0$）

   $$
   (i,t,\ell)\;\longrightarrow\;\mathrm{svc\_in}(i,j,t)
   $$

   只有当：

   * 该服务行驶的能耗 $\Delta \ell^{(\text{srv})}(t,\tau_{ij},d_{ij})$ 可计算，
   * 且 $(i,t,\ell)$ 与 $(j,t+\tau_{ij},\ell-\Delta\ell)$ 都在 reachability 集合 $\mathcal R$ 中（ALL-NEAREST 判据），
   * 且 $\ell\ge \Delta \ell$。

2. **闸门弧** `svc_gate`（**唯一容量弧**，$\tau=0$）

   $$
   \mathrm{svc\_in}(i,j,t)\;\Longrightarrow\;\mathrm{svc\_out}(i,j,t),\qquad
   0\le x \le D_{ijt}
   $$

   * 容量正好等于当期需求 $D_{ijt}$；
   * 费用在 06 里赋成**负成本奖励**：$-\mathrm{VOT}\cdot w_{ijt}$（默认 `unmet_weight_default`，可被 overrides 覆盖）。

3. **退出弧** `svc_exit`（推进时间与扣减 SOC）

   $$
   \mathrm{svc\_out}(i,j,t)\;\longrightarrow\;(j,t+\tau_{ij},\ \ell-\Delta\ell)
   $$

   * $\tau_{ij}=\lceil\frac{\texttt{base\_minutes}_{ij}}{\texttt{dt\_minutes}}\rceil$；
   * $\Delta\ell=\Delta \ell^{(\text{srv})}(t,\tau_{ij},d_{ij})$。

> **关键效果**
>
> * **所有 SOC、所有进入车辆**都必须穿过**同一条** `svc_gate`，这样“满足多少需求”被一条容量弧统一约束；
> * 费用也都**落在 gate 上**（奖励），而不是分摊在 enter/exit 上；
> * `enter/exit` 负责“接入/送回网格 + 处理时间/SOC”。

### ID 与实现细节

* 这两类虚节点的 ID 用 `_pseudo_node_id("svc_in"|"svc_out", i,j,t)` 生成，为**负数**，与网格节点（非负）不冲突；
* `svc_gate` 不是自环：`svc_in` 与 `svc_out` 的伪 ID 不同；建弧后仍有兜底**自环剔除**；
* 三段弧用 `req_key = f"{i}-{j}-{t}"` 彼此关联（便于调试/统计），并统一赋稳定 `arc_id`。

---

## （二）、充电弧的虚节点与转移（q\_in / q\_out）

充电被拆成四段（04 的 `tochg / chg_enter / chg_occ / chg_step`），虚节点对是：

* $\mathrm{q\_in}(k,p)$, $\mathrm{q\_out}(k,p)$（对**站点 $k$**与**充电步时刻 $p$** 的一对伪节点）

其中：

* $k$ 来自每个出发区 $i$ 的**最近站集合** $\mathcal K_i^{\text{near}}$；
* $p$ 是全局时间步（与 $t$ 同一时间轴）。

### 四段弧与状态变化

1. **去站** `tochg`（推进时间 + 扣能）

   $$
   (i,t,\ell)\;\longrightarrow\;(\mathrm{zone}(k),\ t+\tau_{i\to k},\ \ell-\Delta\ell^{(\text{tochg})})
   $$

   * 费用在 06 里按 $\sum\beta_{chg_{p1}}$ 计（$\mathrm{VOT}\cdot\beta_{chg_{p1}}$ 的逐步和）。

2. **进入占位** `chg_enter`（$\tau=0$）

   $$
   (\mathrm{zone}(k), p,\ \ell_p)\;\longrightarrow\;\mathrm{q\_in}(k,p)
   $$

   * $p = t+\tau_{i\to k}$ 是车辆**到站时刻**；
   * 需要到站节点在 $\mathcal R$。

3. **占位（容量弧）** `chg_occ`（$\tau=0$，唯一容量弧）

   $$
   \mathrm{q\_in}(k,p)\;\Longrightarrow\;\mathrm{q\_out}(k,p),\qquad 0\le x \le u_{k,p}
   $$

   * 容量 $u_{k,p}$ 由 `cap_hint` 给定；修改后：

     $$
     u_{k,p}\equiv \hat c_k
       \;=\;\max\!\Bigl\{1,\ \varphi\bigl(\mathrm{plugs}_k\cdot \mathrm{util\_factor}_k\cdot \mathrm{queue\_relax\_factor}\bigr)\Bigr\},
     $$

     其中 $\varphi$ 默认取 $\lfloor\cdot\rfloor$（保守口径）；如缺数据则回退 `default_plugs_per_station`。
   * 费用在 06 里按 $\mathrm{VOT}\cdot\beta_{chg_{p2}}(p)$ 计**每一步的占位成本**。

4. **逐步推进** `chg_step`（推进 1 步时间，SOC 只在最后一步抬升）

   $$
   \mathrm{q\_out}(k,p)\;\longrightarrow\;(\mathrm{zone}(k),\ p+1,\ \ell_{p+1}),
   $$

   * 若 $p$ 是最后一步（第 $\tau_{\text{chg}}-1$ 步），则 $\ell_{p+1}=\ell^\star$；否则 $\ell_{p+1}=\ell_p$；
   * $\tau_{\text{chg}}=\Bigl\lceil \tfrac{\tau_{\text{chg}}(\mathrm{level}(k);\ \ell_{\text{arr}}\!\to\!\ell^\star)}{\Delta t}\Bigr\rceil$ 来自充电曲线（按站功率等级与 SOC 区间查询）；
   * 每个中间网格节点都要求在 $\mathcal R$。

> **关键效果**
>
> * 车辆到站后，必须在**每个充电步 $p$** 先过 `chg_enter`，再穿过**唯一容量弧** `chg_occ`，最后通过 `chg_step` 推进到 $p+1$。
> * **并发上限**完全由 `chg_occ` 控制；**占位成本**也只计在 `chg_occ` 上。
> * 充电 SOC 的提升在**最后一步**一次性完成（当前实现），因此中途步的 `chg_step` 只推进时间不变 SOC。
> * 这条“占用链”强制了**连续占位 $\tau_{\text{chg}}$ 步**（没有“半路离开”的弧）。

### ID 与实现细节

* $\mathrm{q\_in}(k,p), \mathrm{q\_out}(k,p)$ 的 ID 用 `_pseudo_node_id("q_in"|"q_out", k,p)` 生成，亦为负数；
* 每个步 $p$ 都有**一对**虚节点与一条容量弧 `chg_occ`，从而表达**分时并发约束**；
* 容量 $u_{k,p}$ 在当前实现中**对给定站点 $k$ 与所有 $p$ 相同**（若需要分时容量，可把映射做成 $k,p$ 维度）。

---

## （三）、这些虚节点如何与“真实网格”拼接？

* 网格节点是 $(i,t,\ell)$；服务/充电的虚节点只作为**中间枢纽**，不携带供给；
* **流守恒**在所有节点（含虚节点）都成立；

  * 例如对 `svc_gate`：任何通过 gate 的流量必须由一些 `svc_enter` 进入，随后由 `svc_exit` 离开；
  * 对充电：每一步必须“enter → occ → step”按顺序走完，才能推进到下一时间步。
* 在 07 里，为了让求解器输入完整，会把**所有出现在弧端点的虚节点**自动补进节点表，并把其 `supply=0`。
* 连通性裁剪（05）使用 BFS，邻接表基于弧的 `from_node_id`/`to_node_id` 构造，对**负数的虚节点**同样适用。

---

# 例子

## 例 1：只服务，不充电

* 已知：$\tau_{ij}=2$ 步（30min），服务能耗 $\Delta\ell_{\text{srv}}=10$。
* 起点：$(i,\,t,\,\ell)=(i,\,10,\,60)$。

服务三段：

1. **enter**：$(i,10,60)\to \mathrm{svc\_in}(i,j,10)$（$\ell=60\ge 10$，且两端点在 $\mathcal R$）。
2. **gate**（唯一容量弧）：$\mathrm{svc\_in}(i,j,10)\Rightarrow \mathrm{svc\_out}(i,j,10)$，容量 $=D_{i j 10}$，费用为 $-\mathrm{VOT}\cdot w_{i j 10}$。
3. **exit**：$\mathrm{svc\_out}(i,j,10)\to (j,12,50)$（时间 +2，SOC 60→50）。

> 这个例子里没有充电，路径完全由服务三段组成。

---

## 例 2：服务后充电

* 参数：$\tau_{ij}=2,\ \Delta\ell_{\text{srv}}=10;\ \tau_{j\to k}=1,\ \Delta\ell^{(\text{tochg})}=5;\ \tau_{\text{chg}}=3$。
* 站点并发：假设 `plugs=4, util_factor=0.5, queue_relax_factor=1.0`
  $\Rightarrow \hat c_k=\lfloor 4\times 0.5\times 1.0\rfloor=2$（每一步 `chg_occ` 容量=2）。
* 起点：$(i,t,\ell)=(i,\,20,\,80)$。

服务三段：

$$
(i,20,80)\xrightarrow{\text{enter}}\mathrm{svc\_in}(i,j,20)
\Rightarrow^{u=D_{ij20}} \mathrm{svc\_out}(i,j,20)
\xrightarrow{\text{exit}} (j,22,70).
$$

去站 + 占用链四段（目标 $\ell^\star=90$）：

* 去站 `tochg`：$(j,22,70)\to (\mathrm{zone}(k),23,65)$。
* 第 1 充电步 $p=23$：

  * `chg_enter`：$(\mathrm{zone}(k),23,65)\to \mathrm{q\_in}(k,23)$
  * `chg_occ`：$\mathrm{q\_in}(k,23)\Rightarrow \mathrm{q\_out}(k,23)$（容量 $=\hat c_k=2$，费用用 $\mathrm{VOT}\cdot \beta_{chg_{p2}}(23)$）
  * `chg_step`：$\mathrm{q\_out}(k,23)\to (\mathrm{zone}(k),24,65)$（非最后一步，SOC 不变）
* 第 2 步 $p=24$ 同理，推进到 $(\mathrm{zone}(k),25,65)$。
* **第 3（最后）步** $p=25$：

  * `chg_enter` → `chg_occ`（容量 2） → `chg_step(last)`：$(\mathrm{zone}(k),26,\mathbf{90})$。

---

## 例 3：两辆车同时到站、站点容量受限（$\hat c_k=1$ 的“排队”）

* 参数：$\tau_{i\to k}=1,\ \Delta\ell^{(\text{tochg})}=5,\ \tau_{\text{chg}}=2$，站点并发 $\hat c_k=1$。
* 车辆 A、B 同时到站：两者都在 $(\mathrm{zone}(k),\,p=40,\,\ell=55)$。

时间步 $p=40$：

* 两车都**可以**做 `chg_enter` 到 $\mathrm{q\_in}(k,40)$；
* 但唯一容量弧 \`chg\_occ: \mathrm{q\_in}(k,40)\Rightarrow \mathrm{q\_out}(k,40)) 容量 $u_{k,40}=\hat c_k=1$。

  * **A** 走 `chg_occ`（占掉那 1 单位容量），再 `chg_step` 到 $(\mathrm{zone}(k),41,55)$；
  * **B** 如果也想充，就必须等待——**更优的做法**是不进入 $\mathrm{q\_in}$，而是在网格里用**idle 弧**先从 $(\mathrm{zone}(k),40,55)$ 待到 $(\mathrm{zone}(k),41,55)$，再在 $p=41$ 做 `chg_enter`。

时间步 $p=41$（$\tau_{\text{chg}}=2$ 的**最后一步**）：

* A：再次经过 `chg_enter → chg_occ(u=1) → chg_step(last)`，SOC：55→$\ell^\star$（比如 70）；
* B：现在容量空出 1，可以在 $p=41$ 进入并完成它的 **第 1 步**（但它还差最后一步，要到 $p=42$ 才能抬升）。

> 关键点：**并不是模型在 $\mathrm{q\_in}$ 里“排队”**；真正的“排队”效果来自：容量有限时，剩余流量在**网格节点**上多待几步（`idle`），延后再进入下一时刻的 `chg_enter/occ/step`。

---

## 例 4：SOC 不够 → 无法生成去站弧（reachability 拦截）

* 参数：$\tau_{i\to k}=2,\ \Delta\ell^{(\text{tochg})}=25$。
* 起点：$(i,\,t=15,\,\ell=20)$。

由于 $\ell=20 < \Delta\ell^{(\text{tochg})}=25$，且 $(i,20,15)\notin\mathcal R$ 的条件被触发，**`tochg` 弧压根不会被生成**；后续的 `chg_enter/occ/step` 自然也不存在。
此时车辆只能选择：服务（若可行）、重定位（若 $\ell$ ≥ 该弧能耗且在 $\mathcal R$）、或直接 `idle` 等待（SOC 不变、时间 +1），直到有可行动作。

---

## 例 5：跨窗口的“Halo”到达（服务/充电链条可以越过窗末）

* 设滚动窗口 $[t_0,t_{\text{hi}})=[50,\,56)$，Halo $B=3$。
* 服务：某条 $(i\to j,t=55)$ 的服务弧，$\tau_{ij}=2$，则到达时刻 $t+\tau_{ij}=57$；**在窗外但 $\le t_{\text{hi}}+B=59$**，因此 **04 会生成**这条 `svc_exit` 到窗外节点的弧。
* 充电：若去站到达 $p=56$，且 $\tau_{\text{chg}}=3$，链条会产生到 $p=59$ 的三步 `chg_enter/occ/step`；同样因为 $59\le t_{\text{hi}}+B$，**链条完整存在**。
* 05 的连通性裁剪默认 `keep_on="from"`（只要**出发端在窗内**就保留），因此这些跨窗到达的弧**不会被裁掉**。07 会把涉及到的“窗外端点（含虚节点）”也补进节点集，`supply=0`。

---

## 纯服务（不充电、不重定位）

### S1：单车服务（闸门容量=当期需求）

* 参数：$\tau_{ij}=2$ 步（30min），服务能耗 $\Delta\ell_{\text{srv}}=8$，当期需求 $D_{ijt}=3$。
* 起点：$(i,\,t=12,\,\ell=35)$，且 $(i,12,35)$、$(j,14,27)\in\mathcal R$。

三段：

1. `svc_enter`: $(i,12,35) \to \mathrm{svc\_in}(i,j,12)$
2. `svc_gate`（唯一容量弧）：$\mathrm{svc\_in}\Rightarrow \mathrm{svc\_out}$，容量 $=D_{ij,12}=3$，费用为 **负成本** $-\mathrm{VOT}\cdot w_{ij,12}$（默认 $w=1$）
3. `svc_exit`: $\mathrm{svc\_out}(i,j,12) \to (j,14,27)$

> 若没有 `coeff_schedule.csv`，本次服务的**行驶不记费**（费用集中到 gate 的奖励），只有 `svc_gate` 上的 $-1$ 进入目标函数。

---

### S2：两车抢一单（闸门限流）

* 参数：$\tau_{ij}=1$，$\Delta\ell_{\text{srv}}=6$，**当期需求 $D_{ij,20}=1$**（只有 1 单）。
* 两辆车 A、B 都在 $(i,\,20,\,\ell)$，A 的 $\ell=50$，B 的 $\ell=40$；两者都满足 $\ell \ge 6$，且端点在 $\mathcal R$。

在 $t=20$：

* A、B 都可 `svc_enter` 到 $\mathrm{svc\_in}(i,j,20)$；
* 但 `svc_gate` 容量只有 **1**：$\mathrm{svc\_in}\Rightarrow \mathrm{svc\_out}$ 最多放行 1 单。

  * 假设 A 过 gate，B **不能**再过 gate；
  * A 之后 `svc_exit` 到 $(j,21,\ell_A=44)$。
* B 的选择：换服务别的 $j'$；或 `idle` 等下一期；或 `reposition/charging` 等——模型会根据其他弧成本/奖励自行优化。

---

## 重定位（不服务、不充电）

### R1：可行的重定位 +（可选）重定位收益

* 参数：$\tau_{ij}=2$，$\Delta\ell_{\text{rep}}=10$；默认 `min_soc_for_reposition=20`。
* 起点：$(i,\,t=8,\,\ell=50)$；两端点在 $\mathcal R$。

弧与代价：

$$
(i,8,50)\ \xrightarrow{\text{reposition}}\ (j,10,40).
$$

* 可行性：$\ell=50\ge \max(20,10)$；
* 成本：$c^{\text{rep}} = \mathrm{VOT}\cdot\sum_{q=0}^{\tau-1}\gamma_{rep_p}(t+q)$。若 $\gamma_{rep_p}\equiv 1$，则 $=2$。
* **若启用重定位收益**（`enable_reposition_reward=True`，$\gamma_{\text{rep}}>0$）：

  * 06 会算 $\text{zone\_value}(t,j) = \bigl[\sum_{\text{出自 }j}\text{demand}\bigr] - \text{inv0}(j)$ 并截负为 0；
  * $rep_{reward} = -\gamma_{\text{rep}}\cdot \text{zone\_value}(t,j)$（负成本，抵消部分重定位成本）。
    例：$\gamma_{\text{rep}}=10,\ \text{zone\_value}=0.3 \Rightarrow \text{rep\_reward}=-3$，**净成本** $2-3=-1$。

---

### R2：被**时长阈值**剪枝（不生成弧）

* `max_reposition_tt=30` 分钟（默认），$\Delta t=15$。
* 若某 $i\to j$ 的 `base_minutes=45`，则 $\tau=3> \lceil 30/15\rceil=2$ 的阈值，**04 直接不生成**该重定位弧。

---

### R3：被 **SOC 门槛** 拦截（不生成弧）

* `min_soc_for_reposition=20`，某重定位能耗 $\Delta\ell_{\text{rep}}=25$。
* 若起点 $(i,t,\ell=22)$，则 $\ell < \max(20,25)=25$ → **不生成**该重定位弧。

---

## 串联：重定位 → 服务（典型“预定位抢单”）

### RS1：先挪车，再立刻接单

* 参数：重定位 $\tau_{i\to m}=2,\ \Delta\ell_{\text{rep}}=6$；服务 $m\to j$ 的 $\tau_{mj}=1,\ \Delta\ell_{\text{srv}}=5,\ D_{mj,t+2}=2$。
* 起点：$(i,\,t=30,\,\ell=40)$，相关端点在 $\mathcal R$。

路径：

1. `reposition`: $(i,30,40)\to (m,32,34)$（可行：$\ell\ge \max(20,6)$）
2. 立刻服务三段（在 $t=32$）：
   $(m,32,34)\xrightarrow{\text{enter}}\mathrm{svc\_in}(m,j,32)
   \Rightarrow^{u=D_{mj,32}=2}\mathrm{svc\_out}(m,j,32)
   \xrightarrow{\text{exit}}(j,33,29)$

成本：

* 重定位成本 $c^{\text{rep}}=\mathrm{VOT}\cdot(\gamma_{rep_p}(30)+\gamma_{rep_p}(31))$（常数 $\gamma_{rep_p}=1\Rightarrow 2$）；如启用重定位收益则再加上 `rep_reward`；
* 服务奖励：`svc_gate` 上的负成本 $-\mathrm{VOT}\cdot w_{mj,32}$。

> 注意：若 $D_{mj,32}$ 被其他车辆耗尽，这辆车就过不了 gate，需要 `idle` 等下期或改走别的动作。

---

## 纯服务的再补充（更“边界”的两种）

### S3：**可达性**导致服务不可行

* 虽然 $\ell$ 足够覆盖 $\Delta\ell_{\text{srv}}$，但若 $(i,t,\ell)\notin\mathcal R$ 或 $(j,t+\tau,\ell-\Delta\ell)\notin\mathcal R$，**04 不会生成**对应 `svc_enter/exit`（从而也没有那条 `svc_gate` 的可接入路径）。

### S4：**门槛 SOC** 由能耗决定

* 服务不设 `min_soc` 常数门槛，**门槛正是该弧的能耗**：必须 $\ell\ge \Delta\ell_{\text{srv}}(t,\tau,d)$。这点与重定位不同（重定位还有 `min_soc_for_reposition` 的下限）。

