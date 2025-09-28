# 惩罚项（正成本）/ 奖励项（负成本）

> 记号：$\Delta t$ 为时间步长；弧 $a$ 的出发时刻为弧上字段 $t$；$\tau$ 为该弧跨越的步数。
> 正成本=惩罚；负成本=奖励。所有系数最终汇总到弧的 `coef_total`（见表后"汇总口径"）。

## 新增功能特性

### 增强的求解器系统
- **多求解器支持**: GLPK、CBC等求解器自动选择
- **负环检测**: 求解前自动检测负成本环，避免无界解
- **超时控制**: 可配置的求解时间限制
- **性能优化**: 大规模问题的求解优化
- **滚动窗口**: 支持大规模时间范围的滚动优化
- **内存管理**: 优化的内存使用和垃圾回收
- **并行处理**: 支持多窗口并行求解
- **实时输出**: 支持求解过程的实时状态显示

### 可视化分析系统
- **SAEVResultVisualizer**: 完整的求解结果可视化工具
- **多维度分析**: 流量分布、成本效益、时空演化、网络拓扑
- **交互式图表**: 支持plotly、folium等交互式可视化
- **自动化报告**: 生成完整的分析报告
- **实时监控**: 支持滚动窗口求解的实时结果展示
- **地图集成**: 基于地理信息的流量可视化
- **性能仪表板**: 关键运营指标的汇总展示

### 管道执行系统
- **run_pipeline.py**: 支持顺序执行多个脚本步骤
- **run_rolling_solve.py**: 滚动窗口求解的简化接口
- **日志记录**: 完整的执行日志和错误处理
- **重试机制**: 支持失败重试和超时控制
- **回调函数**: 支持窗口完成后的自定义处理
- **实时输出**: 支持求解过程的实时状态显示
- **配置管理**: 统一的配置参数管理

| 项目          | 性质        | 施加在（arc\_type）                                 | 数学表达式（弧系数）                                                                                                               | 开关 / 参数来源                                                                                                                                          | 备注                                                                                                                                             |
| ----------- | --------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **重定位时间成本** | 惩罚        | `reposition`                                   | $\displaystyle \color{#d00}{\text{coef\_rep}} = \mathrm{VOT}\cdot\sum_{q=0}^{\tau-1}\gamma_{rep_p}(t+q)$                         | `coeff_schedule.csv` 中 `gamma_rep_p(t)`；若缺失，用 `cfg.costs_equity.gamma_rep` 常数（默认1.0）；VOT 来自 `cfg.costs_equity.vot`（默认1.0）                                        | 与能耗可行性无关，纯时间惩罚。                                                                              |
| **去充电行驶成本** | 惩罚        | `tochg`                                        | $\displaystyle \color{#d00}{\text{coef\_chg\_travel}} = \mathrm{VOT}\cdot\sum_{q=0}^{\tau_{\text{tochg}}-1}\beta_{chg_{p1}}(t+q)$ | `coeff_schedule.csv` 中 `beta_chg_p1(t)`；缺失用 `cfg.costs_equity.beta_toCHG`（默认1.0）；VOT 同上                                                                      | $\tau_{\text{tochg}}$ 来自站距/速度；仅去站路段计费。                                                                                                         |
| **充电占位成本**  | 惩罚        | `chg_occ`（每步一条）                                | $\displaystyle \color{#d00}{\text{coef\_chg\_occ}} = \mathrm{VOT}\cdot \beta_{chg_{p2}}(t)$                                       | `coeff_schedule.csv` 中 `beta_chg_p2(t)`；缺失用 `cfg.costs_equity.beta_chg`（默认1.0）；VOT 同上                                                                        | 每个占位步各记一次；`chg_occ` 同时是**唯一容量弧**（容量由 07 写入 `capacity`）。                                                                                        |
| **服务奖励**    | 奖励        | `svc_gate`（唯一容量弧）                              | $\displaystyle \color{#090}{\text{coef\_svc\_gate}} = -\,\mathrm{VOT}\cdot w_{ijt}$                                      | 由 06：默认 $w_{ijt}=$`cfg.costs_equity.unmet_weight_default`（默认1.0）；可被 `cfg.unmet_weights_overrides[t][(i,j)]` 覆盖；受 `cfg.flags.enable_service_reward` 控制 | `svc_gate` 的容量 = $D_{ijt}$（当期需求）；奖励以**负成本**落在 gate 上。                                                                                          |
| **重定位收益**   | 奖励        | `reposition`                                   | $\displaystyle \color{#090}{\text{coef\_rep\_reward}} = -\,\gamma_{\text{rep}}\cdot \text{zone\_value}(t,j)$             | $\gamma_{\text{rep}}=$`cfg.costs_equity.gamma_reposition_reward`（默认1.0）；受 `cfg.flags.enable_reposition_reward` 控制                                     | $\text{zone\_value}(t,j)=\bigl[\sum_{\text{出自 }j}\text{demand}(t)\bigr]-\text{inv0}(j)$，取下限 0。`inv0` 来自 `initial_inventory.parquet` 的 $t=t_0$。 |
| **充电收益**    | 奖励        | 优先 `chg_occ`，否则回退 `chg_step`                   | $\displaystyle \color{#090}{\text{coef\_chg\_reward}} = -\,\alpha_{\text{chg}}\cdot(\ell_{\text{to}}-\ell)$              | $\alpha_{\text{chg}}=$`cfg.costs_equity.beta_chg_reward`（默认1.0）；受 `cfg.flags.enable_charging_reward` 控制                                                | 以弧上的 $(\ell,\ell_{\text{to}})$ 计算 $\Delta\text{SOC}$。生成器"**只在最后一步抬升 SOC**"，因此通常由**最后一步**承担该奖励。                                               |
| **汇点弧成本**   | 惩罚（可设为 0） | `to_sink`（07 追加）                               | $\text{cost\_to\_sink}$                                                                                                  | CLI `--cost-to-sink`（默认 0）                                                                                                                         | 用于把窗末节点流汇到超级汇点平衡供给；容量为大数。                                                                                                                      |
| **其它弧**     | 0 成本      | `idle`, `svc_enter`, `svc_exit`, `chg_enter` 等 | $0$                                                                                                                      | ——                                                                                                                                                 | 这些弧只负责拓扑连通、时间推进或能耗扣减（在状态上体现），不直接记费。                                                                                                            |

> **容量口径**
>
> * `svc_gate`：容量 = $D_{ijt}$（来自 OD）。
> * `chg_occ`：容量 = $\hat c_{k,p}$。改动后由 `plugs × util_factor × queue_relax_factor` 的保守取整得到（06 不改容量，只计每步占位成本；07 把 `cap_hint` 写进 `capacity` 字段）。
> * 其它弧：07 统一写入“无穷容量”= `--cap-infinite`（默认 $1\text{e}{12}$）。

---

## 成本**汇总**与落盘口径（07）

06 会把各项系数合并到窗口弧表，07 用 `cost = coef_total` 写入 `arcs.parquet`。各弧的净成本为：

* `reposition`：$\textbf{coef\_total} = \text{coef\_rep} + \text{coef\_rep\_reward}$
* `tochg`：$\textbf{coef\_total} = \text{coef\_chg\_travel}$
* `chg_occ`：$\textbf{coef\_total} = \text{coef\_chg\_occ} + \text{coef\_chg\_reward}$（若奖励落在 `chg_step`，则本项只剩 $\text{coef\_chg\_occ}$）
* `chg_step`：$\textbf{coef\_total} = \text{coef\_chg\_reward}$（若奖励未落在 `chg_step`，则为 0）
* `svc_gate`：$\textbf{coef\_total} = \text{coef\_svc\_gate}$
* `idle / svc_enter / svc_exit / chg_enter`：$\textbf{coef\_total}=0$
* `to_sink`：$\textbf{coef\_total}= \text{cost\_to\_sink}$（默认 0）

> 保护措施：若出现 **`svc_gate` 自环**（理论不该有），06 会把该 gate 的奖励清零（`coef_svc_gate=0`）并进而使 `coef_total=0`。

# 优化目标

把已经由 02–07 生成好的**网络图**当成一个标准的**单商品最小费用流**问题来解：

$$
\min \sum_{a\in\mathcal A} c_a\,x_a
\quad\text{s.t.}\quad
\begin{cases}
\sum_{a:u(a)=v} x_a-\sum_{a:v(a)=v}x_a=b_v, & \forall v\in\mathcal V \\
0 \le x_a \le U_a, & \forall a\in\mathcal A
\end{cases}
$$

* $x_a$：弧 $a$ 上的车辆流量（连续变量）。
* $c_a$：该弧的单位成本（已在 06/07 阶段把“时间成本/占位成本/奖励”等全部合成到这里；奖励就是**负成本**）。
* $U_a$：该弧容量（`svc_gate`=需求上限、`chg_occ`=站点并发上限，其他弧多为大数）。
* $b_v$：节点供给（正=源，负=汇）。07 通常会加**超级汇点**把总供需配平。

从优化目标看，本脚本要做的就是：
**在所有合法的流量分配中，找一组流 $x$**，既满足所有容量与供需平衡，又让**总成本最小**。
这会自然地同时权衡“去服务/去充电/去重定位/空等”的取舍，因为这些选择都已经被编码成不同成本的弧。

---

# 工作原理

可以把脚本理解为一个"**表格 → 线性规划 → 求解**"的流水线：

1. **读图**

   * `nodes` 表：`node_id, supply`。
   * `arcs` 表：`from_node_id, to_node_id, cost, capacity`（还有可选 `arc_type, arc_id` 等仅用于汇总）。
   * 路径自动发现：优先 `--nodes/--arcs`，否则读 `meta.json`，再兜底找 `data/solver_graph/*`。

2. **健康检查与清洗**

   * 强制转型（ID→int64、cost/capacity/supply→float）；`capacity` 缺失填**大数**。
   * 仅保留**被弧连接**或**有非零供需**的节点，减少模型规模。
   * **供需平衡检查**：$\sum b_v=0$ 不成立就报错（提示建图时加超级汇点）。

3. **负环检测（新增）**

   * 求解前自动检测负成本环，避免无界解。
   * 使用 `cfg.solver.check_negative_cycles` 控制（默认True）。
   * 发现负环时给出警告或错误信息。

4. **自动生成线性规划（LP）**

   * **每条弧**建一个连续变量 $x_a\in[0,U_a]$。
   * **每个节点**建一条"流出−流入=供给"的等式约束。
   * **目标**是 $\sum c_a x_a$ 最小。

5. **选择求解器并求解**

   * 默认 `auto`：优先 GLPK，失败回落 CBC。
   * 支持超时控制：`cfg.solver.time_limit_sec`（默认1800秒）。
   * 这是**纯 LP**（没有整数变量），因此速度快、可扩展性好。

6. **回写与汇总**

   * 把每条弧的最优流量写回 `flow` 列；$|flow|<\text{zero-tol}$ 的置 0。
   * 输出 `solve_summary.json`：最优值、总成本、各 `arc_type` 的流量/成本汇总等。

> 直觉类比：
> 这等价于在一张**时间展开图**里，寻找**若干条“从源到汇”的多条路径组合**，每条边有价格（成本）与容量限制；
> 求解器会自动在“**更便宜**但可能**受限**”和“**更贵**但**不拥塞**”的路径之间做权衡。

---

# 冲突与竞争如何被“自动处理”

* **充电并发受限**：`chg_occ` 弧的容量就是“每步最多能多少车并发占位”。如果多车同时想充，**总和不能超过容量**；超出的会被迫排到后续时间步，或去其他站，或改做别的弧（哪条更便宜，LP 就选哪条）。
* **服务闸门限流**：`svc_gate` 弧的容量是该 (i,j,t) 的最大可服务量（需求上限）。若供给不足，闸门弧不会满流；若供给充足但有其他更优选择（例如区内别的高价值需求），也可能不会把闸门打满。
* **资源抢占**：当“两个选择互斥”时，最终谁“抢到”资源，取决于全网成本最小化的总目标（而不是局部贪心）；这就是 LP 在做的全局折中。

---

# 为什么只需要“看数字”，却能做“业务决策”

* 脚本\*\*不理解“时间、SOC、站点”\*\*这些业务概念；
* 这些概念在 02–07 已经被**编码成图**：

  * 时间推进 = 弧方向；
  * SOC 改变 = 节点坐标变化；
  * 需求上限 = `svc_gate` 容量；
  * 充电并发 = `chg_occ` 容量；
  * 激励与惩罚 = 弧成本（正/负）；
* 因此求解阶段只需解一个**干净的 MCF**，就能“顺便”实现所有业务规则。

---

# 会得到什么样的解

* **每条弧的最优流量**（连续）：表示“在该时间步、该动作”的派量。
* **总成本**：是把“时间成本 + 占位成本 − 奖励”等都加总后的**净目标**。
* **类型分解**：知道充电/服务/重定位分别走了多少流、花了多少成本。

若需要“整数车数”，可把变量改成 `Integer`（规模和时间会显著上升）。在当前连续版中，“2.4 辆车”理解为**期望或可分派的份额**，对于大规模滚动调度是常见做法。

---

## 一句话总结

**目标**：在容量与供需平衡下，使总成本 $\sum c_a x_a$ 最小。
**原理**：把"时间—空间—SOC—容量—奖励"都预先编码成有向图上的**成本/容量/供需**，再解一个**标准的 LP 版最小费用流**。
求解器因此能自动处理"充电并发冲突""闸门限流""去服务/去充/去重定位的权衡"等所有竞争关系，并给出全局最优的流量分配。

## 新增功能总结

### 可视化分析
- **SAEVResultVisualizer**: 提供完整的求解结果可视化
- **多维度分析**: 流量分布、成本效益、时空演化、网络拓扑
- **交互式图表**: 支持plotly、folium等现代可视化技术
- **自动化报告**: 生成详细的分析报告
- **实时监控**: 支持滚动窗口求解的实时结果展示
- **地图集成**: 基于地理信息的流量可视化
- **性能仪表板**: 关键运营指标的汇总展示

### 管道执行
- **run_pipeline.py**: 支持顺序执行多个处理步骤
- **run_rolling_solve.py**: 滚动窗口求解的简化接口
- **日志管理**: 完整的执行日志和错误处理
- **重试机制**: 支持失败重试和超时控制
- **回调函数**: 支持窗口完成后的自定义处理
- **实时输出**: 支持求解过程的实时状态显示
- **配置管理**: 统一的配置参数管理

### 求解器增强
- **负环检测**: 求解前自动检测负成本环
- **多求解器支持**: GLPK、CBC等求解器自动选择
- **性能优化**: 大规模问题的求解优化
- **滚动窗口**: 支持大规模时间范围的滚动优化
- **内存管理**: 优化的内存使用和垃圾回收
- **并行处理**: 支持多窗口并行求解
- **实时输出**: 支持求解过程的实时状态显示
