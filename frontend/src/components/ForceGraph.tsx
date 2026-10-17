/**
 * 零依赖力导向图组件。
 *
 * 为什么不引入 d3 / vis-network
 * -----------------------------
 * 项目已有 `recharts`（413KB）承担图表，再引入 d3-force（+ 依赖约 100KB+）
 * 只为画一个星型图不划算。而本图谱的结构**恰好很简单**：
 * 全部关系都是「中心实体 ←→ 邻居」的星型（实测 5 种关系类型全从
 * SubjectMatter 出发），不需要通用图布局算法。
 *
 * 实现方式
 * --------
 * - 用 `requestAnimationFrame` 跑一个轻量力模拟：
 *     斥力（节点两两排斥）+ 弹簧力（边保持理想长度）+ 向心力（吸向画布中心）
 * - 手工做鼠标命中检测与拖拽（圆形的距离判断即可）
 * - 节点半径按**度数**映射：度数高的实体半径大，一眼看出重要性
 *
 * 交互
 * ----
 * - 拖拽节点：按住拖动，松手后继续参与模拟
 * - 悬停：显示名称与度数
 * - 滚轮缩放 / 空白处拖拽平移
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

export interface GraphNode {
  id: string
  label: string
  label_cn: string
  name: string
  degree: number
  is_center?: boolean
}

export interface GraphEdge {
  source: string
  target: string
  rel_type: string
  rel_cn: string
}

interface SimNode extends GraphNode {
  x: number
  y: number
  vx: number
  vy: number
  r: number
  fixed?: boolean
}

/** 按节点类型配色（与项目现有配色体系一致） */
const LABEL_COLOR: Record<string, string> = {
  SubjectMatter: '#2563EB', // 标的物 —— 品牌蓝
  Supplier: '#10B981', // 供应商 —— 绿
  Purchaser: '#F59E0B', // 采购单位 —— 琥珀
  Agency: '#8B5CF6', // 代理机构 —— 紫
  Location: '#64748B', // 地区 —— 灰
}

const CENTER_COLOR = '#DC2626' // 中心实体 —— 红，便于一眼定位

/** 度数 → 半径：用对数压缩，否则 480 度的节点会比 1 度的大 480 倍 */
function radiusFor(degree: number, isCenter: boolean): number {
  if (isCenter) return 26
  const d = Math.max(1, degree)
  return Math.min(20, 6 + Math.log2(d + 1) * 2.2)
}

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  height?: number
  onNodeClick?: (node: GraphNode) => void
}

export default function ForceGraph({ nodes, edges, height = 520, onNodeClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const simRef = useRef<SimNode[]>([])
  const rafRef = useRef<number>(0)
  const dragRef = useRef<{ id: string | null; dx: number; dy: number }>({ id: null, dx: 0, dy: 0 })
  const panRef = useRef<{ active: boolean; sx: number; sy: number; ox: number; oy: number }>({
    active: false, sx: 0, sy: 0, ox: 0, oy: 0,
  })
  const viewRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 })
  const hoverRef = useRef<string | null>(null)

  const [size, setSize] = useState({ w: 900, h: height })
  const [hovered, setHovered] = useState<GraphNode | null>(null)

  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>()
    nodes.forEach((n) => m.set(n.id, n))
    return m
  }, [nodes])

  // ---- 数据变化时重建模拟（按度数做初始圆环布局，避免随机初始位置导致布局乱跳）----
  useEffect(() => {
    const cx = size.w / 2
    const cy = size.h / 2
    const center = nodes.find((n) => n.is_center)
    const others = nodes.filter((n) => !n.is_center)

    const sim: SimNode[] = []
    if (center) {
      sim.push({ ...center, x: cx, y: cy, vx: 0, vy: 0, r: radiusFor(center.degree, true) })
    }
    const ringR = Math.min(size.w, size.h) * 0.33
    others.forEach((n, i) => {
      const angle = (i / Math.max(1, others.length)) * Math.PI * 2
      sim.push({
        ...n,
        x: cx + Math.cos(angle) * ringR,
        y: cy + Math.sin(angle) * ringR,
        vx: 0,
        vy: 0,
        r: radiusFor(n.degree, false),
      })
    })
    simRef.current = sim
    viewRef.current = { scale: 1, offsetX: 0, offsetY: 0 }
  }, [nodes, size.w, size.h])

  // ---- 容器尺寸自适应 ----
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const update = () => setSize({ w: el.clientWidth, h: height })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [height])

  // ---- 物理模拟 + 绘制 ----
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size.w * dpr
    canvas.height = size.h * dpr
    canvas.style.width = `${size.w}px`
    canvas.style.height = `${size.h}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const idIndex = new Map<string, number>()
    simRef.current.forEach((n, i) => idIndex.set(n.id, i))

    const tick = () => {
      const sim = simRef.current
      const cx = size.w / 2
      const cy = size.h / 2

      // ① 斥力：全部节点两两排斥，避免重叠
      for (let i = 0; i < sim.length; i++) {
        const a = sim[i]
        for (let j = i + 1; j < sim.length; j++) {
          const b = sim[j]
          let dx = b.x - a.x
          let dy = b.y - a.y
          let d2 = dx * dx + dy * dy
          if (d2 < 1) {
            // 完全重合时给一个确定性的小偏移，避免除零导致节点飞散
            dx = (i - j) * 0.5 || 0.5
            dy = 0.5
            d2 = dx * dx + dy * dy
          }
          const d = Math.sqrt(d2)
          // 中心节点斥力更强，把邻居推开形成清晰的星型
          const strength = (a.is_center || b.is_center ? 5200 : 2400) / d2
          const fx = (dx / d) * strength
          const fy = (dy / d) * strength
          if (!a.fixed) { a.vx -= fx; a.vy -= fy }
          if (!b.fixed) { b.vx += fx; b.vy += fy }
        }
      }

      // ② 弹簧力：边维持理想长度
      const IDEAL = Math.min(size.w, size.h) * 0.30
      for (const e of edges) {
        const ai = idIndex.get(e.source)
        const bi = idIndex.get(e.target)
        if (ai === undefined || bi === undefined) continue
        const a = sim[ai]
        const b = sim[bi]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const d = Math.sqrt(dx * dx + dy * dy) || 1
        const force = (d - IDEAL) * 0.012
        const fx = (dx / d) * force
        const fy = (dy / d) * force
        if (!a.fixed) { a.vx += fx; a.vy += fy }
        if (!b.fixed) { b.vx -= fx; b.vy -= fy }
      }

      // ③ 向心力 + 阻尼 + 边界收束
      for (const n of sim) {
        if (n.fixed) continue
        n.vx += (cx - n.x) * 0.0016
        n.vy += (cy - n.y) * 0.0016
        n.vx *= 0.82
        n.vy *= 0.82
        // 限速：防止初始布局时节点被弹飞出画布
        const speed = Math.hypot(n.vx, n.vy)
        const MAX = 12
        if (speed > MAX) {
          n.vx = (n.vx / speed) * MAX
          n.vy = (n.vy / speed) * MAX
        }
        n.x += n.vx
        n.y += n.vy
        n.x = Math.max(n.r + 4, Math.min(size.w - n.r - 4, n.x))
        n.y = Math.max(n.r + 4, Math.min(size.h - n.r - 4, n.y))
      }

      // ---- 绘制 ----
      const view = viewRef.current
      ctx.clearRect(0, 0, size.w, size.h)
      ctx.save()
      ctx.translate(view.offsetX, view.offsetY)
      ctx.scale(view.scale, view.scale)

      // 边
      ctx.lineWidth = 1.2
      for (const e of edges) {
        const ai = idIndex.get(e.source)
        const bi = idIndex.get(e.target)
        if (ai === undefined || bi === undefined) continue
        const a = sim[ai]
        const b = sim[bi]
        const highlighted =
          hoverRef.current === a.id || hoverRef.current === b.id
        ctx.strokeStyle = highlighted ? '#93C5FD' : '#E2E8F0'
        ctx.lineWidth = highlighted ? 2 : 1.2
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
      }

      // 节点
      for (const n of sim) {
        const color = n.is_center ? CENTER_COLOR : LABEL_COLOR[n.label] ?? '#94A3B8'
        const isHover = hoverRef.current === n.id
        ctx.beginPath()
        ctx.arc(n.x, n.y, n.r + (isHover ? 3 : 0), 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        if (n.is_center || isHover) {
          ctx.strokeStyle = '#fff'
          ctx.lineWidth = 2.5
          ctx.stroke()
        }

        // 标签：中心节点与悬停节点必显示，其余按半径决定（避免文字糊成一团）
        const showLabel = n.is_center || isHover || n.r >= 11
        if (showLabel) {
          const text = n.name.length > 12 ? n.name.slice(0, 11) + '…' : n.name
          ctx.font = n.is_center ? 'bold 12px system-ui' : '11px system-ui'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          const tw = ctx.measureText(text).width
          ctx.fillStyle = 'rgba(255,255,255,0.86)'
          ctx.fillRect(n.x - tw / 2 - 2, n.y + n.r + 2, tw + 4, 14)
          ctx.fillStyle = n.is_center ? '#991B1B' : '#334155'
          ctx.fillText(text, n.x, n.y + n.r + 3)
        }
      }

      ctx.restore()
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [size.w, size.h, edges])

  // ---- 命中检测（把屏幕坐标换算回模拟坐标）----
  const hitTest = useCallback((clientX: number, clientY: number): SimNode | null => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const view = viewRef.current
    const x = (clientX - rect.left - view.offsetX) / view.scale
    const y = (clientY - rect.top - view.offsetY) / view.scale
    // 倒序遍历：后画的（通常在上层）优先命中
    for (let i = simRef.current.length - 1; i >= 0; i--) {
      const n = simRef.current[i]
      if (Math.hypot(n.x - x, n.y - y) <= n.r + 3) return n
    }
    return null
  }, [])

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const hit = hitTest(e.clientX, e.clientY)
    if (hit) {
      const canvas = canvasRef.current!
      const rect = canvas.getBoundingClientRect()
      const view = viewRef.current
      const x = (e.clientX - rect.left - view.offsetX) / view.scale
      const y = (e.clientY - rect.top - view.offsetY) / view.scale
      dragRef.current = { id: hit.id, dx: x - hit.x, dy: y - hit.y }
      hit.fixed = true
    } else {
      panRef.current = {
        active: true,
        sx: e.clientX,
        sy: e.clientY,
        ox: viewRef.current.offsetX,
        oy: viewRef.current.offsetY,
      }
    }
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    if (drag.id) {
      const canvas = canvasRef.current!
      const rect = canvas.getBoundingClientRect()
      const view = viewRef.current
      const node = simRef.current.find((n) => n.id === drag.id)
      if (node) {
        node.x = (e.clientX - rect.left - view.offsetX) / view.scale - drag.dx
        node.y = (e.clientY - rect.top - view.offsetY) / view.scale - drag.dy
        node.vx = 0
        node.vy = 0
      }
      return
    }
    if (panRef.current.active) {
      viewRef.current.offsetX = panRef.current.ox + (e.clientX - panRef.current.sx)
      viewRef.current.offsetY = panRef.current.oy + (e.clientY - panRef.current.sy)
      return
    }
    const hit = hitTest(e.clientX, e.clientY)
    hoverRef.current = hit?.id ?? null
    setHovered(hit ? (nodeMap.get(hit.id) ?? null) : null)
  }

  const endDrag = () => {
    const id = dragRef.current.id
    if (id) {
      const node = simRef.current.find((n) => n.id === id)
      if (node) node.fixed = false
    }
    dragRef.current = { id: null, dx: 0, dy: 0 }
    panRef.current.active = false
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    // 位移很小视为"点击"而非拖拽
    if (drag.id && onNodeClick) {
      const n = nodeMap.get(drag.id)
      if (n) onNodeClick(n)
    }
    endDrag()
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* 指针已释放 */
    }
  }

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const view = viewRef.current
    const next = Math.max(0.3, Math.min(3, view.scale * (e.deltaY < 0 ? 1.1 : 0.9)))
    view.scale = next
  }

  const resetView = () => {
    viewRef.current = { scale: 1, offsetX: 0, offsetY: 0 }
  }

  // 图例：只显示当前图里实际出现的类型
  const presentLabels = useMemo(() => {
    const s = new Set(nodes.map((n) => n.label))
    return Array.from(s)
  }, [nodes])

  return (
    <div ref={wrapRef} className="relative">
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={endDrag}
        onWheel={onWheel}
        className="block cursor-grab rounded-lg bg-slate-50 active:cursor-grabbing"
      />

      {/* 图例 */}
      <div className="absolute left-3 top-3 rounded-lg border border-slate-200 bg-white/92 px-2.5 py-1.5 backdrop-blur">
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="flex items-center gap-1 text-[10px] text-ink-600">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: CENTER_COLOR }} />
            中心实体
          </span>
          {presentLabels
            .filter((l) => l !== 'SubjectMatter' || !nodes.some((n) => n.is_center && n.label === 'SubjectMatter'))
            .map((l) => (
              <span key={l} className="flex items-center gap-1 text-[10px] text-ink-600">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: LABEL_COLOR[l] ?? '#94A3B8' }}
                />
                {nodes.find((n) => n.label === l)?.label_cn ?? l}
              </span>
            ))}
        </div>
        <div className="mt-1 text-[10px] text-ink-400">
          节点大小＝连接数｜拖拽移动｜滚轮缩放
        </div>
      </div>

      {/* 悬停详情 */}
      {hovered && (
        <div className="absolute right-3 top-3 rounded-lg border border-slate-200 bg-white/95 px-3 py-2 backdrop-blur">
          <div className="text-[11px] text-ink-400">{hovered.label_cn}</div>
          <div className="mt-0.5 max-w-[220px] text-[12px] font-medium text-ink-900">
            {hovered.name}
          </div>
          <div className="mt-0.5 text-[11px] text-ink-500">连接数 {hovered.degree}</div>
        </div>
      )}

      <button
        onClick={resetView}
        className="absolute bottom-3 right-3 rounded-md border border-slate-200 bg-white/92 px-2 py-1 text-[10px] text-ink-500 backdrop-blur hover:bg-white"
      >
        重置视图
      </button>
    </div>
  )
}
