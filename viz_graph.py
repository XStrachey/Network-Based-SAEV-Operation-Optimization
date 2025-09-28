# -*- coding: utf-8 -*-
"""
viz_graph.py — 交互式网络图（有向通路 trace + 动态重排）
功能：
- t → 列；zone → 横带；列内网格布局
- 虚节点：按相邻边类型分 charging / service / unknown，并在各 (t, zone) 中“分排”
- flow-only：一键仅显示 flow>0 的边并隐藏不参与的节点
- focus on click：点击节点仅显示其一度邻域并动态重排（左入右出，中下双向）
- trace path (flow)：点击节点，仅保留其在 **有向 flow>0** 边上的通路，并按层级动态重排
  * path dir: forward / backward / both
用法：
  python viz_graph.py -i simple_graph.json -o graph.html
"""
import argparse, collections, json, math
from pathlib import Path
from string import Template

# ---------- helpers ----------
def is_num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def canon_id(val):
    s = str(val)
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return str(v)
    except Exception:
        return s

def fmt_num(val):
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(f)
    except Exception:
        return str(val)

def get_uv(edge):
    for s_key in ("source", "from", "u", "src"):
        for t_key in ("target", "to", "v", "dst"):
            if s_key in edge and t_key in edge:
                return canon_id(edge[s_key]), canon_id(edge[t_key])
    return None, None

def norm_zone(z):
    if z is None:
        return "(NA)"
    try:
        if isinstance(z, float) and math.isnan(z):
            return "(NA)"
    except Exception:
        pass
    return str(z)

def norm_type(x):
    return str(x).strip().lower() if x is not None else ""

def parse_set_arg(s):
    return {norm_type(t) for t in s.split(",") if t.strip()}

def parse_list_arg(s):
    return [norm_type(t) for t in s.split(",") if t.strip()]

# ---------- HTML ----------
HTML_TMPL = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Graph by t (directed trace + relayout)</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<style>
  html, body { height: 100%; margin: 0; }
  .row {
    padding: 8px 12px; border-bottom: 1px solid #e5e7eb;
    display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  #legend, #vlegend { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .chip { display: inline-flex; gap: 6px; align-items: center;
          padding: 2px 8px; border-radius: 999px; background: #f3f4f6; font-size: 12px; }
  .swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  #network { width: 100%; height: calc(100% - 168px); }
  label { user-select: none; }
  .btn { padding: 4px 10px; border:1px solid #e5e7eb; background:#fff; border-radius:8px; cursor:pointer; }
  div.vis-network div.vis-tooltip { white-space: pre-line; max-width: 520px; }
</style>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <div class="row">
    <div><strong>t columns:</strong> ${TCOLS}</div>
    <div><strong>zones:</strong> ${ZONES}</div>
    <label><input type="checkbox" id="toggleLabels"> show labels</label>
    <label title="只保留 flow>0 的边，并隐藏未参与这些边的节点">
      <input type="checkbox" id="flowOnly"> flow-only
    </label>
    <label><input type="checkbox" id="showVirtual" checked> show virtual nodes</label>
  </div>

  <div class="row">
    <div id="legend"><strong>Edge types:</strong>${EDGE_CHIPS}</div>
  </div>

  <div class="row">
    <div id="vlegend"><strong>Virtual categories:</strong>${VIRTUAL_CHIPS}</div>
  </div>

  <div class="row">
    <label title="点击节点后，仅显示该节点与其所有直连边及对端节点，并动态重排">
      <input type="checkbox" id="focusOnClick"> focus on click
    </label>
    <button class="btn" id="clearFocus" title="退出聚焦并恢复原始布局">clear</button>

    <label title="点击节点后，仅保留其在 flow>0 的有向通路，并按层级动态重排">
      <input type="checkbox" id="tracePath"> trace path (flow)
    </label>
    <span>
      path dir:
      <label><input type="radio" name="pathdir" value="forward"> forward</label>
      <label><input type="radio" name="pathdir" value="backward"> backward</label>
      <label><input type="radio" name="pathdir" value="both" checked> both</label>
    </span>
    <button class="btn" id="clearPath" title="关闭通路模式并恢复筛选">clear path</button>
  </div>

  <div id="network"></div>

  <script>
    const ALL_NODES = ${NODES_JSON};
    const ALL_EDGES = ${EDGES_JSON};
    const FOCUS = { xGap: ${FOCUS_X}, yGap: ${FOCUS_Y}, maxCols: ${FOCUS_MAX_COLS}, centerSize: 11 };
    const PATH  = { xGap: ${PATH_X},  yGap: ${PATH_Y},  maxCols: ${PATH_MAX_COLS}, centerSize: 11 };

    const nodes = new vis.DataSet(ALL_NODES);
    let edges = new vis.DataSet(ALL_EDGES);

    const options = {
      physics: false,
      interaction: { hover: true, zoomView: true, dragView: true },
      nodes: { shape: 'dot', font: { size: 1 } },
      edges: { smooth: false, arrows: 'to' }
    };
    const network = new vis.Network(document.getElementById('network'), { nodes, edges }, options);

    // 保存原始坐标用于恢复
    const ORIG_POS = {};
    nodes.get().forEach(n => { ORIG_POS[n.id] = { x: n.x, y: n.y, size: n.size ?? 8 }; });

    // 状态
    let focusedNodeId = null;
    let tracedCenterId = null;

    // 标签切换
    document.getElementById('toggleLabels').addEventListener('change', e => {
      const show = e.target.checked;
      nodes.update(nodes.get().map(n => ({ id: n.id, label: show ? String(n.id) : ' ' })));
    });

    function getPathDir() {
      const el = document.querySelector('input[name="pathdir"]:checked');
      return el ? el.value : 'both';
    }

    // 点击优先级：trace path > focus
    network.on('click', params => {
      if (!(params.nodes && params.nodes.length > 0)) return;
      const nid = String(params.nodes[0]);
      if (document.getElementById('tracePath').checked) {
        tracedCenterId = nid; focusedNodeId = null; applyFilters(); return;
      }
      if (document.getElementById('focusOnClick').checked) {
        focusedNodeId = nid; tracedCenterId = null; applyFilters();
      }
    });
    document.getElementById('clearFocus').addEventListener('click', () => { focusedNodeId = null; applyFilters(); });
    document.getElementById('clearPath').addEventListener('click', () => { tracedCenterId = null; applyFilters(); });
    document.querySelectorAll('input[name="pathdir"]').forEach(r => r.addEventListener('change', () => {
      if (tracedCenterId !== null && document.getElementById('tracePath').checked) applyFilters();
    }));

    function restoreOriginalLayout() {
      const upd = nodes.get().map(n => ({
        id: n.id, x: ORIG_POS[n.id]?.x, y: ORIG_POS[n.id]?.y,
        size: ORIG_POS[n.id]?.size ?? n.size, fixed: true, hidden: n.hidden
      }));
      nodes.update(upd);
    }

    // 网格摆放
    function gridPlace(list, baseX, baseY, gapX, gapY, maxCols) {
      const n = list.length; if (n === 0) return {};
      const cols = Math.min(maxCols, Math.max(1, Math.ceil(Math.sqrt(n))));
      const rows = Math.ceil(n / cols);
      const pos = {};
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / cols), c = i % cols;
        const x = baseX + (c - (cols - 1) / 2) * (gapX / 2);
        const y = baseY + (r - (rows - 1) / 2) * gapY;
        pos[list[i]] = { x, y };
      }
      return pos;
    }

    // ======= 有向通路（flow>0） + 动态重排 =======
    function showDirectedPathRelayout(centerId, dir) {
      // 1) 仅 flow>0 的有向邻接
      const outAdj = new Map(), inAdj = new Map();
      const posEdges = [];
      for (const e of ALL_EDGES) {
        if (!(e.flow > 0)) continue;
        posEdges.push(e);
        if (!outAdj.has(e.from)) outAdj.set(e.from, []);
        if (!inAdj.has(e.to))    inAdj.set(e.to, []);
        outAdj.get(e.from).push(e.to);
        inAdj.get(e.to).push(e.from);
      }

      // 2) BFS 距离
      function bfs(start, adjMap) {
        const dist = new Map([[start, 0]]), q = [start];
        while (q.length) {
          const u = q.shift();
          for (const v of (adjMap.get(u) || [])) {
            if (!dist.has(v)) { dist.set(v, dist.get(u) + 1); q.push(v); }
          }
        }
        return dist;
      }
      const distF = (dir === 'forward' || dir === 'both') ? bfs(centerId, outAdj) : new Map([[centerId,0]]);
      const distB = (dir === 'backward' || dir === 'both') ? bfs(centerId, inAdj ) : new Map([[centerId,0]]);

      // 3) 选节点并分层（左负右正）
      const keep = new Set([centerId]);
      for (const k of distF.keys()) keep.add(k);
      for (const k of distB.keys()) keep.add(k);

      const layer = new Map([[centerId, 0]]);
      keep.forEach(nid => {
        if (nid === centerId) return;
        const inF = distF.has(nid), inB = distB.has(nid);
        if (dir === 'forward' && inF) layer.set(nid, +distF.get(nid));
        else if (dir === 'backward' && inB) layer.set(nid, -distB.get(nid));
        else if (dir === 'both') {
          if (inF && !inB) layer.set(nid, +distF.get(nid));
          else if (!inF && inB) layer.set(nid, -distB.get(nid));
          else if (inF && inB) layer.set(nid, (distF.get(nid) <= distB.get(nid)) ? +distF.get(nid) : -distB.get(nid));
        }
      });

      // 4) 布局与显示
      const byLayer = new Map();
      for (const nid of keep) {
        const lv = layer.get(nid); if (lv === undefined) continue;
        if (!byLayer.has(lv)) byLayer.set(lv, []);
        byLayer.get(lv).push(nid);
      }
      nodes.update(nodes.get().map(n => ({ id: n.id, hidden: !keep.has(n.id) })));

      const updates = [{ id: centerId, x: 0, y: 0, size: PATH.centerSize, fixed: true }];
      for (const [lv, arr] of Array.from(byLayer.entries()).sort((a,b)=>a[0]-b[0])) {
        if (lv === 0) continue;
        const baseX = lv * PATH.xGap, baseY = 0;
        const pos = gridPlace(arr, baseX, baseY, PATH.xGap, PATH.yGap, PATH.maxCols);
        for (const [id,p] of Object.entries(pos)) updates.push({ id, x:p.x, y:p.y, fixed:true });
      }
      nodes.update(updates);

      // 5) 仅显示通路上的 flow>0 边
      const keepEdges = [];
      const seen = new Set();
      for (const e of posEdges) {
        if (keep.has(e.from) && keep.has(e.to)) {
          const key = e.from + "->" + e.to + "|" + e.edge_type + "|" + (e.cost ?? "");
          if (!seen.has(key)) { seen.add(key); keepEdges.push({ ...e, hidden: false }); }
        }
      }
      edges.clear(); edges.add(keepEdges);

      // 视图对齐
      network.fit({ nodes: Array.from(keep), animation: { duration: 250, easingFunction: "easeInOutQuad" } });
    }

    // ======= 聚焦动态重排 =======
    function relayoutFocus(centerId) {
      const inSet  = new Set();
      const outSet = new Set();
      ALL_EDGES.forEach(e => { if (e.to === centerId && !inSet.has(e.from)) inSet.add(e.from);
                               if (e.from === centerId && !outSet.has(e.to)) outSet.add(e.to); });
      const biSet = new Set();
      for (const n of Array.from(inSet)) if (outSet.has(n)) { biSet.add(n); inSet.delete(n); outSet.delete(n); }

      const inList  = Array.from(inSet);
      const outList = Array.from(outSet);
      const biList  = Array.from(biSet);

      const keep = new Set([centerId, ...inList, ...outList, ...biList]);
      nodes.update(nodes.get().map(n => ({ id: n.id, hidden: !keep.has(n.id) })));

      const updates = [{ id: centerId, x: 0, y: 0, size: FOCUS.centerSize, fixed: true }];
      const posIn  = gridPlace(inList,  -FOCUS.xGap, 0, FOCUS.xGap, FOCUS.yGap, FOCUS.maxCols);
      const posOut = gridPlace(outList, +FOCUS.xGap, 0, FOCUS.xGap, FOCUS.yGap, FOCUS.maxCols);
      const posBi  = gridPlace(biList,  0, FOCUS.yGap*1.4, FOCUS.xGap, FOCUS.yGap, FOCUS.maxCols);
      for (const [id,p] of Object.entries(posIn))  updates.push({ id, x:p.x, y:p.y, fixed:true });
      for (const [id,p] of Object.entries(posOut)) updates.push({ id, x:p.x, y:p.y, fixed:true });
      for (const [id,p] of Object.entries(posBi))  updates.push({ id, x:p.x, y:p.y, fixed:true });
      nodes.update(updates);

      const focusEdges = ALL_EDGES.filter(e => e.from === centerId || e.to === centerId).map(e => ({ ...e, hidden: false }));
      edges.clear(); edges.add(focusEdges);
      network.fit({ nodes: Array.from(keep), animation: { duration: 250, easingFunction: "easeInOutQuad" } });
    }

    // 统一过滤
    function applyFilters() {
      // 1) 通路模式（有向 + 动态重排）
      if (tracedCenterId !== null && document.getElementById('tracePath').checked) {
        restoreOriginalLayout();  // 先恢复坐标，再重排通路
        showDirectedPathRelayout(tracedCenterId, getPathDir());
        return;
      }
      // 2) 聚焦模式
      if (focusedNodeId !== null && document.getElementById('focusOnClick').checked) {
        relayoutFocus(focusedNodeId);
        return;
      }

      // 3) 常规筛选
      const allowedEdgeTypes = new Set(Array.from(document.querySelectorAll('.etype')).filter(b => b.checked).map(b => b.value));
      const allowedVCats    = new Set(Array.from(document.querySelectorAll('.vcat')).filter(b => b.checked).map(b => b.value));
      const flowOnly        = document.getElementById('flowOnly').checked;
      const showVirtual     = document.getElementById('showVirtual').checked;

      const hiddenInit = new Set();
      nodes.get().forEach(n => {
        if (n.is_virtual) {
          const ok = showVirtual && allowedVCats.has(n.vcat);
          if (!ok) hiddenInit.add(n.id);
        }
      });

      const candidateEdges = ALL_EDGES.filter(e =>
        allowedEdgeTypes.has(e.edge_type) && (!flowOnly || e.flow > 0)
      );

      const activeNodeIds = new Set();
      candidateEdges.forEach(e => {
        if (!hiddenInit.has(e.from) && !hiddenInit.has(e.to)) {
          activeNodeIds.add(e.from); activeNodeIds.add(e.to);
        }
      });

      const finalHidden = new Set(hiddenInit);
      if (flowOnly) {
        nodes.get().forEach(n => {
          if (!finalHidden.has(n.id) && !activeNodeIds.has(n.id)) finalHidden.add(n.id);
        });
      }

      restoreOriginalLayout();
      nodes.update(nodes.get().map(n => ({ id: n.id, hidden: finalHidden.has(n.id) })));

      const filtered = ALL_EDGES.map(e => ({
        ...e,
        hidden: !(
          allowedEdgeTypes.has(e.edge_type) &&
          (!flowOnly || e.flow > 0) &&
          !finalHidden.has(e.from) &&
          !finalHidden.has(e.to)
        )
      }));
      edges.clear(); edges.add(filtered);
    }

    // 常规控件
    document.getElementById('flowOnly').addEventListener('change', applyFilters);
    document.getElementById('showVirtual').addEventListener('change', applyFilters);
    document.querySelectorAll('.etype').forEach(b => b.addEventListener('change', applyFilters));
    document.querySelectorAll('.vcat').forEach(b => b.addEventListener('change', applyFilters));
  </script>
</body>
</html>
""")

# ---------- 生成可视化数据 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="simple_graph.json")
    ap.add_argument("-o", "--output", default="graph.html")
    # 虚节点分类依据的边类型
    ap.add_argument("--charging-types", default="tochg,chg_enter,chg_occ,chg_step")
    ap.add_argument("--service-types",  default="svc_enter,svc_gate,svc_exit")
    # 虚节点“分排”
    ap.add_argument("--virtual-lane-order", default="charging,service,unknown")
    ap.add_argument("--virtual-lane-gap", type=float, default=140)
    ap.add_argument("--virtual-column", choices=["left","right"], default="left")
    # 全局布局
    ap.add_argument("--x-gap", type=float, default=260)
    ap.add_argument("--zone-gap", type=float, default=420)
    ap.add_argument("--grid-x", type=float, default=26)
    ap.add_argument("--grid-y", type=float, default=26)
    ap.add_argument("--max-cols", type=int, default=9)
    # 聚焦布局参数
    ap.add_argument("--focus-x-gap", type=float, default=260)
    ap.add_argument("--focus-y-gap", type=float, default=110)
    ap.add_argument("--focus-max-cols", type=int, default=6)
    # 通路布局参数
    ap.add_argument("--path-x-gap", type=float, default=260, help="通路模式：相邻层的水平间距")
    ap.add_argument("--path-y-gap", type=float, default=110, help="通路模式：层内网格的垂直间距")
    ap.add_argument("--path-max-cols", type=int, default=7, help="通路模式：层内网格最大列数")
    args = ap.parse_args()

    CHARGING = parse_set_arg(args.charging_types)
    SERVICE  = parse_set_arg(args.service_types)

    V_LANE_ORDER = parse_list_arg(args.virtual_lane_order)
    seen = set(); order = []
    for c in V_LANE_ORDER + ["unknown"]:
        if c and c not in seen:
            seen.add(c); order.append(c)
    V_LANE_ORDER = order
    lane_index = {c:i+1 for i,c in enumerate(V_LANE_ORDER)}  # 0 给“实际节点”

    data = json.load(open(args.input, "r", encoding="utf-8"))
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    # 规范化节点
    nodes_all = []
    for n in raw_nodes:
        m = dict(n)
        m["id"] = canon_id(n.get("id"))
        m["zone"] = norm_zone(n.get("zone"))
        nodes_all.append(m)

    actual_nodes = [n for n in nodes_all if is_num(n.get("t"))]
    virtual_nodes = [n for n in nodes_all if not is_num(n.get("t"))]
    if not actual_nodes:
        raise SystemExit("未找到任何数值 t 的节点，无法按 t 排列。")

    t_vals = sorted({float(n["t"]) for n in actual_nodes})
    t_to_idx = {t: i for i, t in enumerate(t_vals)}
    zones = sorted({n.get("zone") for n in nodes_all})
    z_to_idx = {z: i for i, z in enumerate(zones)}

    # 邻接与 incident edges
    id_to_t = {n["id"]: float(n["t"]) for n in actual_nodes}
    neighbors = collections.defaultdict(set)
    edges_by_node = collections.defaultdict(list)
    for e in raw_edges:
        u, v = get_uv(e)
        if not u or not v:
            continue
        neighbors[u].add(v); neighbors[v].add(u)
        edges_by_node[u].append(e); edges_by_node[v].append(e)

    def snap_to_nearest_t(ts):
        target = sum(ts) / len(ts)
        return min(t_vals, key=lambda tv: abs(tv - target))

    virtual_to_t = {}
    for vn in virtual_nodes:
        vs = [id_to_t[nid] for nid in neighbors.get(vn["id"], []) if nid in id_to_t]
        virtual_to_t[vn["id"]] = snap_to_nearest_t(vs) if vs else None

    # 虚节点类别（看相邻边类型）
    def classify_virtual(node_id):
        types = [norm_type(e.get("type")) for e in edges_by_node.get(node_id, [])]
        c = sum(1 for t in types if t in CHARGING)
        s = sum(1 for t in types if t in SERVICE)
        if c > s and c > 0: return "charging"
        if s > c and s > 0: return "service"
        if c == s and c > 0: return "charging"
        return "unknown"

    for vn in virtual_nodes:
        vn["_vcat"] = classify_virtual(vn["id"])

    # 分组 (col, zone, is_virtual, vcat)
    groups = collections.defaultdict(list)
    for n in actual_nodes:
        groups[(t_to_idx[float(n["t"])], n["zone"], False, "")].append(n)
    for n in virtual_nodes:
        vt = virtual_to_t[n["id"]]
        if vt is None:
            col = -1 if args.virtual_column == "left" else len(t_vals)
        else:
            col = t_to_idx[vt]
        groups[(col, n["zone"], True, n["_vcat"])].append(n)

    EDGE_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                   "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    VCAT_COLORS = { "charging": "#06b6d4", "service": "#ef4444", "unknown": "#9ca3af" }

    # 节点布局
    XG, YG, GX, GY, MAXC = args.x_gap, args.zone_gap, args.grid_x, args.grid_y, args.max_cols
    vis_nodes = []
    for (col, zone, is_virtual, vcat), arr in groups.items():
        x_center = col * XG
        if col < 0: x_center = -XG
        if col >= len(t_vals): x_center = len(t_vals) * XG

        base = -z_to_idx.get(zone, 0) * YG
        y_base = base if not is_virtual else base + (lane_index.get(vcat, lane_index.get("unknown", 1)) * args.virtual_lane_gap)

        nsize = len(arr)
        cols = min(MAXC, max(1, int(math.ceil(math.sqrt(nsize)))))
        rows = int(math.ceil(nsize / cols))

        for i, nd in enumerate(arr):
            r, c = divmod(i, cols)
            x = x_center + (c - (cols - 1) / 2.0) * GX
            y = y_base + (r - (rows - 1) / 2.0) * GY

            lines = [f"Node: {nd['id']}",
                     f"t: {'virtual' if is_virtual and not is_num(nd.get('t')) else fmt_num(nd.get('t'))}",
                     f"zone: {zone}"]
            if is_virtual: lines.append(f"virtual category: {vcat}")
            if "soc" in nd: lines.append(f"soc: {fmt_num(nd['soc'])}")
            if "supply" in nd: lines.append(f"supply: {fmt_num(nd['supply'])}")
            title = "\n".join(lines)

            base_node = {
                "id": nd["id"], "label": " ", "title": title,
                "x": x, "y": y, "fixed": True,
                "group": f"zone_{zone}",
                "is_virtual": bool(is_virtual),
                "vcat": vcat if is_virtual else ""
            }
            if is_virtual:
                color = VCAT_COLORS.get(vcat, "#9ca3af")
                base_node.update({"shape":"diamond","size":8,"color":{"background":color,"border":color}})
            else:
                base_node.update({"shape":"dot","size":8})
            vis_nodes.append(base_node)

    # 边
    etypes, vis_edges_all = [], []
    id_set_all = {n["id"] for n in nodes_all}
    virtual_id_set = {n["id"] for n in virtual_nodes}
    for e in raw_edges:
        u, v = get_uv(e)
        if not u or not v:
            continue
        if u not in id_set_all or v not in id_set_all:
            continue
        et = e.get("type", "edge"); etypes.append(et)
        flow_raw = e.get("flow", 0.0)
        try:
            flow = float(flow_raw)
        except:
            flow = 0.0
        width = max(1, min(6, 1 + int(abs(flow) ** 0.5)))
        lines = [f"Edge: {u} → {v}", f"type: {et}"]
        if "flow" in e: lines.append(f"flow: {fmt_num(flow_raw)}")
        if "cost" in e: lines.append(f"cost: {fmt_num(e['cost'])}")
        vis_edges_all.append({
            "from": u, "to": v, "arrows": "to",
            "width": width, "smooth": False,
            "title": "\n".join(lines),
            "edge_type": et, "flow": flow,
            "dashes": True if (u in virtual_id_set or v in virtual_id_set) else False
        })

    etypes_sorted = sorted(set(etypes))
    edge_type_to_color = {t: EDGE_COLORS[i % len(EDGE_COLORS)] for i, t in enumerate(etypes_sorted)}
    for e in vis_edges_all:
        e["color"] = {"color": edge_type_to_color.get(e["edge_type"], "#7f7f7f")}

    def chip(label, color, cls, value):
        return (f'<label class="chip">'
                f'<span class="swatch" style="background:{color}"></span>'
                f'<input type="checkbox" class="{cls}" value="{value}" checked/> {label}'
                f'</label>')

    edge_chips = "".join(chip(t, edge_type_to_color[t], "etype", t) for t in etypes_sorted)

    order = parse_list_arg(args.virtual_lane_order)
    if "unknown" not in order:
        order.append("unknown")
    vcat_colors = { "charging": "#06b6d4", "service": "#ef4444", "unknown": "#9ca3af" }
    vcat_chips = "".join(chip(vc, vcat_colors[vc], "vcat", vc) for vc in order)

    tcols_text = ", ".join(fmt_num(t) for t in t_vals)
    zones_text = ", ".join(str(z) for z in zones)

    html = HTML_TMPL.substitute(
        TCOLS=tcols_text, ZONES=zones_text,
        EDGE_CHIPS=edge_chips, VIRTUAL_CHIPS=vcat_chips,
        NODES_JSON=json.dumps(vis_nodes, ensure_ascii=False),
        EDGES_JSON=json.dumps(vis_edges_all, ensure_ascii=False),
        FOCUS_X=str(int(args.focus_x_gap)),
        FOCUS_Y=str(int(args.focus_y_gap)),
        FOCUS_MAX_COLS=str(int(args.focus_max_cols)),
        PATH_X=str(int(args.path_x_gap)),
        PATH_Y=str(int(args.path_y_gap)),
        PATH_MAX_COLS=str(int(args.path_max_cols)),
    )
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"[OK] wrote: {args.output}")

if __name__ == "__main__":
    main()
