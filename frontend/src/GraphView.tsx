import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import cytoscape from "cytoscape";
import {
  Card,
  Select,
  Segmented,
  Slider,
  Space,
  Typography,
  Tag,
  Spin,
  Input,
  Button,
  message,
  InputNumber,
  Switch,
  Modal,
  List,
  Tabs,
  Badge,
  Divider,
  Drawer,
  Tooltip,
  Popover
} from "antd";
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  FilterOutlined,
  AppstoreOutlined,
  SyncOutlined,
  AimOutlined,
  DeleteOutlined,
  SettingOutlined,
  LeftOutlined,
  RightOutlined
} from "@ant-design/icons";
import { jsPDF } from "jspdf";
import {
  fetchGraph,
  fetchNeighbors,
  fetchRecommendations,
  exportPath,
  matchZotero,
  pushZotero,
  updatePaper,
  fetchSubfields,
  createSubfield,
  updateSubfield,
  deleteSubfield,
  fetchSyncStatus,
  enqueueAllSync,
  runSync,
  cleanupCitations,
  zoteroMatchAll,
  zoteroPushAll,
  zoteroSyncIds,
  backfillPapers,
  fetchPaperNotes,
  savePaperNotes,
  fetchPaperBacklinks,
  fetchPaperSchema,
  extractPaperSchema,
  searchSchemas,
  fetchTasks,
  createTask,
  updateTask,
  completeReview,
  fetchExperiments,
  createExperiment,
  deleteExperiment,
  fetchShortestPath,
  comparePapers,
  classifyCitationIntent,
  chatWithPapers,
  streamChatWithPapers,
  createChatSession,
  fetchChatMessages,
  fetchChatSessions,
  fetchIdeaCapsuleBoard,
  fetchIdeaCapsules,
  createIdeaCapsule,
  updateIdeaCapsule,
  deleteIdeaCapsule,
  fetchJobAlerts,
  fetchJobHistory,
  resolveJobAlert,
  retryJob,
  runQueuedJobs,
  fetchMetricProvenance,
  fetchSchemaOntology,
  alignSchemaOntology,
  discoverOpenTags,
  fetchMetricLeaderboard,
  searchPapers,
  type GraphNode,
  type GraphResponse,
  type Subfield,
  type SyncStatusResponse,
  type PaperNote,
  type PaperSchema,
  type BacklinksResponse,
  type ReadingTask,
  type Experiment,
  type ComparePapersResponse,
  type ChatWithPapersResponse,
  type MetricLeaderboard,
  type MetricProvenance,
  type IdeaCapsule,
  type JobAlert,
  type JobRun,
  type ChatMessage,
  type ChatSession
} from "./api";

const { Title, Text, Paragraph } = Typography;

type GraphViewProps = {
  t: (key: string) => string;
  standalone?: boolean;
};

type ClusterHullShape = {
  id: string;
  label: string;
  fill: string;
  stroke: string;
  path: string;
  labelX: number;
  labelY: number;
  size: number;
};

type EdgeBundlePath = {
  id: string;
  path: string;
  stroke: string;
  width: number;
  opacity: number;
  count: number;
  labelX: number;
  labelY: number;
};

export default function GraphView({ t, standalone = false }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [zoteroKey, setZoteroKey] = useState<string>("");
  const [subField, setSubField] = useState<string | undefined>();
  const [yearRange, setYearRange] = useState<[number, number] | null>(null);
  const [strategy, setStrategy] = useState<"foundation" | "sota" | "cluster" | undefined>();
  const [layout, setLayout] = useState<"force" | "hierarchy" | "timeline">("force");
  const [sizeBy, setSizeBy] = useState<"citations" | "pagerank">("citations");
  const [edgeRecentYears, setEdgeRecentYears] = useState<number | undefined>();
  const [minIndegree, setMinIndegree] = useState<number | undefined>();
  const [lazyMode, setLazyMode] = useState(false);
  const [nodeLimit, setNodeLimit] = useState(200);
  const [sortBy, setSortBy] = useState<"citation_count" | "year" | "pagerank">("citation_count");
  const [uploadedOnly, setUploadedOnly] = useState(true);
  const [edgeFocus, setEdgeFocus] = useState<"all" | "out" | "in">("all");
  const [focusMode, setFocusMode] = useState(false);
  const [smartDeclutter, setSmartDeclutter] = useState(true);
  const [edgeIntentFilter, setEdgeIntentFilter] = useState<
    "all" | "build_on" | "contrast" | "use_as_baseline" | "mention"
  >("all");
  const [showLabels, setShowLabels] = useState(false);
  const [labelMode, setLabelMode] = useState<"auto" | "selected" | "all">("auto");
  const hoveredNodeIdRef = useRef<string | null>(null);
  const showLabelsRef = useRef(showLabels);
  const labelModeRef = useRef(labelMode);
  const selectedNodeIdRef = useRef<string | null>(null);
  const baseColors = useRef<Record<string, string>>({});
  const [highlightOrder, setHighlightOrder] = useState<string[]>([]);
  const animationRef = useRef<number[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [subfields, setSubfields] = useState<Subfield[]>([]);
  const [allSubfields, setAllSubfields] = useState<Subfield[]>([]);
  const [subfieldModalOpen, setSubfieldModalOpen] = useState(false);
  const [newSubfieldName, setNewSubfieldName] = useState("");
  const [openTagLoading, setOpenTagLoading] = useState(false);
  const [openTagCandidates, setOpenTagCandidates] = useState<string[]>([]);
  const [syncStatus, setSyncStatus] = useState<SyncStatusResponse>({});
  const [syncLimit, setSyncLimit] = useState(5);
  const [syncBusy, setSyncBusy] = useState(false);
  const [zoteroLimit, setZoteroLimit] = useState(20);
  const [autoPush, setAutoPush] = useState(false);
  const [backfillLimit, setBackfillLimit] = useState(5);
  const [edgeMinConfidence, setEdgeMinConfidence] = useState<number>(0);
  const [sidebarTab, setSidebarTab] = useState<"filters" | "legend" | "sync">("filters");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [sidebarDrawerOpen, setSidebarDrawerOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<"details" | "chat">("details");
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [readingMode, setReadingMode] = useState(false);
  const [showClusterHull, setShowClusterHull] = useState(true);
  const [edgeBundling, setEdgeBundling] = useState(true);
  const [bundledPaths, setBundledPaths] = useState<EdgeBundlePath[]>([]);
  const [clusterHulls, setClusterHulls] = useState<ClusterHullShape[]>([]);
  const [communityCount, setCommunityCount] = useState(0);
  const [timelineGuides, setTimelineGuides] = useState<Array<{ year: number; x: number }>>([]);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [note, setNote] = useState<PaperNote | null>(null);
  const [noteSaving, setNoteSaving] = useState(false);
  const [backlinks, setBacklinks] = useState<BacklinksResponse | null>(null);
  const [paperSchema, setPaperSchema] = useState<PaperSchema | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaKeyword, setSchemaKeyword] = useState("");
  const [schemaSearchResults, setSchemaSearchResults] = useState<
    Array<{
      paper_id: number;
      title?: string;
      year?: number;
      sub_field?: string;
      event_types: Array<{ name: string; roles?: string[] }>;
      role_types: string[];
    }>
  >([]);
  const [metricLeaderboard, setMetricLeaderboard] = useState<MetricLeaderboard | null>(null);
  const [metricType, setMetricType] = useState<"f1" | "precision" | "recall" | "trigger_f1" | "argument_f1">(
    "f1"
  );
  const [tasks, setTasks] = useState<ReadingTask[]>([]);
  const [taskTitle, setTaskTitle] = useState("");
  const [taskDueDate, setTaskDueDate] = useState<string | undefined>();
  const [taskLoading, setTaskLoading] = useState(false);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [expName, setExpName] = useState("");
  const [expModel, setExpModel] = useState("");
  const [expMetrics, setExpMetrics] = useState("");
  const [expDataset, setExpDataset] = useState("");
  const [expSplit, setExpSplit] = useState("");
  const [expMetricName, setExpMetricName] = useState("F1");
  const [expMetricValue, setExpMetricValue] = useState<number | undefined>();
  const [expIsSota, setExpIsSota] = useState(false);
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [compareQueue, setCompareQueue] = useState<number[]>([]);
  const [compareData, setCompareData] = useState<ComparePapersResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [pathSourceId, setPathSourceId] = useState<number | undefined>();
  const [pathTargetId, setPathTargetId] = useState<number | undefined>();
  const [pathLoading, setPathLoading] = useState(false);
  const [qaQuery, setQaQuery] = useState("");
  const [qaMode, setQaMode] = useState<"single" | "compare">("single");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaResult, setQaResult] = useState<ChatWithPapersResponse | null>(null);
  const [chatSessionId, setChatSessionId] = useState<number | undefined>();
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [workbenchTab, setWorkbenchTab] = useState<
    | "chat"
    | "search"
    | "notes"
    | "schema"
    | "tasks"
    | "backlinks"
    | "experiments"
    | "leaderboard"
    | "comparison"
    | "capsules"
    | "jobs"
  >("chat");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<
    Array<{
      paper: { id: number; title?: string | null; year?: number | null; sub_field?: string | null };
      score: number;
      bm25_score: number;
      semantic_score: number;
      snippet?: string;
    }>
  >([]);
  const [metricProvenance, setMetricProvenance] = useState<MetricProvenance | null>(null);
  const [schemaOntology, setSchemaOntology] = useState<{
    nodes: Array<{ id: number; name: string; type: string; aliases: string[]; paper_count: number }>;
    edges: Array<{ source: number; target: number; co_count: number }>;
    count: number;
  } | null>(null);
  const [capsules, setCapsules] = useState<IdeaCapsule[]>([]);
  const [capsuleTitle, setCapsuleTitle] = useState("");
  const [capsuleContent, setCapsuleContent] = useState("");
  const [capsuleStatus, setCapsuleStatus] = useState<"seed" | "incubating" | "validated" | "archived">("seed");
  const [capsulePriority, setCapsulePriority] = useState(2);
  const [capsuleTags, setCapsuleTags] = useState("");
  const [capsuleLoading, setCapsuleLoading] = useState(false);
  const [capsuleBoard, setCapsuleBoard] = useState<Record<string, IdeaCapsule[]>>({});
  const [jobRuns, setJobRuns] = useState<JobRun[]>([]);
  const [jobAlerts, setJobAlerts] = useState<JobAlert[]>([]);
  const [jobLoading, setJobLoading] = useState(false);
  const [menu, setMenu] = useState<{ visible: boolean; x: number; y: number; node: GraphNode | null }>({
    visible: false,
    x: 0,
    y: 0,
    node: null
  });
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const timelinePositions = useRef<{ year: number; x: number }[]>([]);
  const dragStartX = useRef<number | null>(null);
  const dragAnchorYear = useRef<number | null>(null);
  const communityByNodeRef = useRef<Record<string, string>>({});
  const communityMetaRef = useRef<Record<string, { label: string; color: string; stroke: string; size: number }>>({});
  const hullRafRef = useRef<number | null>(null);
  const bundleRafRef = useRef<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragRange, setDragRange] = useState<[number, number] | null>(null);
  const [pendingRange, setPendingRange] = useState<[number, number] | null>(null);

  const fetchData = async (nextSub?: string, nextRange?: [number, number] | null) => {
    setLoading(true);
    try {
      const params: {
        sub_field?: string;
        year_from?: number;
        year_to?: number;
        size_by?: "citations" | "pagerank";
        edge_recent_years?: number;
        min_indegree?: number;
        limit?: number;
        offset?: number;
        sort_by?: "citation_count" | "year" | "pagerank";
        uploaded_only?: boolean;
        edge_min_confidence?: number;
      } = {};
      if (nextSub) params.sub_field = nextSub;
      if (nextRange) {
        params.year_from = nextRange[0];
        params.year_to = nextRange[1];
      }
      params.uploaded_only = uploadedOnly;
      params.size_by = sizeBy;
      params.sort_by = sortBy;
      if (edgeRecentYears) params.edge_recent_years = edgeRecentYears;
      if (minIndegree) params.min_indegree = minIndegree;
      if (edgeMinConfidence > 0) params.edge_min_confidence = edgeMinConfidence;
      if (lazyMode) params.limit = nodeLimit;
      const data = await fetchGraph(params);
      setGraph(data);
      if (data.nodes.length <= 30) {
        setShowLabels(true);
        setLabelMode("all");
      } else if (data.nodes.length <= 120) {
        setShowLabels(true);
        setLabelMode("auto");
      } else {
        setShowLabels(false);
        setLabelMode("selected");
      }
      const minAll = data.meta.year_min_all ?? data.meta.year_min;
      const maxAll = data.meta.year_max_all ?? data.meta.year_max;
      if (!yearRange && minAll && maxAll) {
        setYearRange([minAll, maxAll]);
      }
    } finally {
      setLoading(false);
    }
  };

  const communityPalette = [
    { fill: "rgba(99, 102, 241, 0.005)", stroke: "rgba(79, 70, 229, 0.15)" },
    { fill: "rgba(20, 184, 166, 0.005)", stroke: "rgba(15, 118, 110, 0.15)" },
    { fill: "rgba(245, 158, 11, 0.005)", stroke: "rgba(180, 83, 9, 0.15)" },
    { fill: "rgba(239, 68, 68, 0.005)", stroke: "rgba(185, 28, 28, 0.15)" },
    { fill: "rgba(168, 85, 247, 0.005)", stroke: "rgba(126, 34, 206, 0.15)" },
    { fill: "rgba(14, 165, 233, 0.005)", stroke: "rgba(3, 105, 161, 0.15)" }
  ];

  const timelinePadding = {
    left: 84,
    right: 84,
    top: 150,
    bottom: 118
  };

  const convexHull = (points: Array<{ x: number; y: number }>) => {
    if (points.length <= 1) return points;
    const sorted = points
      .slice()
      .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
    const cross = (
      o: { x: number; y: number },
      a: { x: number; y: number },
      b: { x: number; y: number }
    ) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower: Array<{ x: number; y: number }> = [];
    for (const point of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
        lower.pop();
      }
      lower.push(point);
    }
    const upper: Array<{ x: number; y: number }> = [];
    for (let idx = sorted.length - 1; idx >= 0; idx -= 1) {
      const point = sorted[idx];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
        upper.pop();
      }
      upper.push(point);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper);
  };

  const pathFromPoints = (points: Array<{ x: number; y: number }>) => {
    if (!points.length) return "";
    const hull = convexHull(points);
    if (!hull.length) return "";
    if (hull.length === 1) {
      const p = hull[0];
      const r = 30;
      return `M ${p.x - r} ${p.y} a ${r} ${r} 0 1 0 ${r * 2} 0 a ${r} ${r} 0 1 0 ${-r * 2} 0`;
    }
    if (hull.length === 2) {
      const [a, b] = hull;
      const minX = Math.min(a.x, b.x) - 22;
      const minY = Math.min(a.y, b.y) - 22;
      const maxX = Math.max(a.x, b.x) + 22;
      const maxY = Math.max(a.y, b.y) + 22;
      return `M ${minX + 12} ${minY} L ${maxX - 12} ${minY} Q ${maxX} ${minY} ${maxX} ${minY + 12} L ${maxX} ${maxY - 12} Q ${maxX} ${maxY} ${maxX - 12} ${maxY} L ${minX + 12} ${maxY} Q ${minX} ${maxY} ${minX} ${maxY - 12} L ${minX} ${minY + 12} Q ${minX} ${minY} ${minX + 12} ${minY} Z`;
    }
    const centroid = hull.reduce(
      (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
      { x: 0, y: 0 }
    );
    centroid.x /= hull.length;
    centroid.y /= hull.length;
    const expanded = hull.map((point) => {
      const dx = point.x - centroid.x;
      const dy = point.y - centroid.y;
      const len = Math.hypot(dx, dy) || 1;
      const pad = 16;
      return { x: point.x + (dx / len) * pad, y: point.y + (dy / len) * pad };
    });
    return `M ${expanded.map((point) => `${point.x} ${point.y}`).join(" L ")} Z`;
  };

  const detectCommunities = () => {
    const cy = cyRef.current;
    if (!cy) {
      communityByNodeRef.current = {};
      communityMetaRef.current = {};
      setCommunityCount(0);
      return;
    }
    const nodeIds = cy.nodes().map((node) => node.id());
    if (!nodeIds.length) {
      communityByNodeRef.current = {};
      communityMetaRef.current = {};
      setCommunityCount(0);
      return;
    }
    const neighbors = new Map<string, string[]>();
    nodeIds.forEach((id) => neighbors.set(id, []));
    cy.edges().forEach((edge) => {
      const source = edge.source().id();
      const target = edge.target().id();
      neighbors.get(source)?.push(target);
      neighbors.get(target)?.push(source);
    });
    const labels = new Map<string, string>();
    nodeIds.forEach((id) => labels.set(id, id));
    const orderedNodes = nodeIds
      .slice()
      .sort((a, b) => (neighbors.get(b)?.length ?? 0) - (neighbors.get(a)?.length ?? 0));
    for (let iter = 0; iter < 18; iter += 1) {
      let changed = 0;
      for (const id of orderedNodes) {
        const nbs = neighbors.get(id) ?? [];
        if (!nbs.length) continue;
        const score = new Map<string, number>();
        nbs.forEach((nb) => {
          const label = labels.get(nb);
          if (!label) return;
          score.set(label, (score.get(label) ?? 0) + 1);
        });
        const best = Array.from(score.entries()).sort((a, b) => {
          if (b[1] !== a[1]) return b[1] - a[1];
          return a[0].localeCompare(b[0]);
        })[0]?.[0];
        if (best && best !== labels.get(id)) {
          labels.set(id, best);
          changed += 1;
        }
      }
      if (!changed) break;
    }
    const groups = new Map<string, string[]>();
    labels.forEach((label, nodeId) => {
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label)?.push(nodeId);
    });
    const ranked = Array.from(groups.entries()).sort((a, b) => b[1].length - a[1].length);
    const nodeToCommunity: Record<string, string> = {};
    const meta: Record<string, { label: string; color: string; stroke: string; size: number }> = {};
    ranked.forEach(([groupId, ids], idx) => {
      const key = `c${idx + 1}`;
      ids.forEach((id) => {
        nodeToCommunity[id] = key;
      });
      const topSubField = ids
        .map((id) => cy.$id(id).data("full")?.sub_field || cy.$id(id).data("full")?.open_sub_field)
        .filter(Boolean)
        .reduce<Record<string, number>>((acc, field) => {
          if (!field) return acc;
          acc[field] = (acc[field] ?? 0) + 1;
          return acc;
        }, {});
      const topLabel =
        Object.entries(topSubField).sort((a, b) => b[1] - a[1])[0]?.[0] ??
        `${t("graph.community")} ${idx + 1}`;
      const palette = communityPalette[idx % communityPalette.length];
      meta[key] = { label: topLabel, color: palette.fill, stroke: palette.stroke, size: ids.length };
      void groupId;
    });
    communityByNodeRef.current = nodeToCommunity;
    communityMetaRef.current = meta;
    setCommunityCount(ranked.length);
  };

  const refreshClusterHulls = () => {
    const cy = cyRef.current;
    if (!cy) {
      setClusterHulls([]);
      setTimelineGuides([]);
      return;
    }
    syncCanvasSize();
    if (layout === "timeline") {
      const yearMap = new Map<number, number[]>();
      cy.nodes().forEach((node) => {
        const year = Number(node.data("full")?.year ?? 0);
        if (!year) return;
        if (!yearMap.has(year)) yearMap.set(year, []);
        yearMap.get(year)?.push(node.renderedPosition().x);
      });
      const width = containerRef.current?.clientWidth ?? 800;
      const allYears = Array.from(yearMap.keys()).sort((a, b) => a - b);
      const minYear = allYears[0] ?? 2000;
      const maxYear = allYears[allYears.length - 1] ?? 2026;
      const span = Math.max(1, maxYear - minYear);
      const guideRows = allYears.map((year) => ({
        year,
        x:
          timelinePadding.left +
          ((year - minYear) / span) * (width - timelinePadding.left - timelinePadding.right)
      }));
      setTimelineGuides(guideRows);
    } else {
      setTimelineGuides([]);
    }

    if (!showClusterHull) {
      setClusterHulls([]);
      return;
    }

    const grouped = new Map<string, string[]>();
    Object.entries(communityByNodeRef.current).forEach(([nodeId, clusterId]) => {
      if (!grouped.has(clusterId)) grouped.set(clusterId, []);
      grouped.get(clusterId)?.push(nodeId);
    });
    const hulls: ClusterHullShape[] = [];
    grouped.forEach((nodeIds, clusterId) => {
      if (nodeIds.length < 3) return;
      const points: Array<{ x: number; y: number }> = [];
      let centerX = 0;
      let centerY = 0;
      let count = 0;
      nodeIds.forEach((nodeId) => {
        const node = cy.$id(nodeId);
        if (node.empty()) return;
        const rendered = node.renderedPosition();
        const radius = Math.max(12, (node.renderedWidth() || 16) / 2 + 12);
        points.push(
          { x: rendered.x - radius, y: rendered.y },
          { x: rendered.x + radius, y: rendered.y },
          { x: rendered.x, y: rendered.y - radius },
          { x: rendered.x, y: rendered.y + radius }
        );
        centerX += rendered.x;
        centerY += rendered.y;
        count += 1;
      });
      if (count < 2 || points.length < 2) return;
      const meta = communityMetaRef.current[clusterId];
      hulls.push({
        id: clusterId,
        label: meta?.label || `${t("graph.community")} ${clusterId}`,
        fill: meta?.color || "rgba(99, 102, 241, 0.008)",
        stroke: meta?.stroke || "rgba(99, 102, 241, 0.15)",
        path: pathFromPoints(points),
        labelX: centerX / count,
        labelY: centerY / count,
        size: meta?.size || count
      });
    });
    setClusterHulls(hulls);
  };

  const scheduleHullRefresh = () => {
    if (hullRafRef.current !== null) return;
    hullRafRef.current = window.requestAnimationFrame(() => {
      hullRafRef.current = null;
      refreshClusterHulls();
    });
  };

  const syncCanvasSize = () => {
    const container = containerRef.current;
    const width = container?.clientWidth || 800;
    const height = container?.clientHeight || 600;
    setCanvasSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }));
  };

  const refreshEdgeBundles = () => {
    const cy = cyRef.current;
    if (!cy) {
      setBundledPaths([]);
      return;
    }
    syncCanvasSize();

    cy.edges().removeClass("bundled-edge");
    if (!edgeBundling) {
      setBundledPaths([]);
      return;
    }

    const visibleEdges = cy
      .edges()
      .filter((edge) => !edge.hasClass("focus-dim") && !edge.hasClass("declutter-edge"));
    if (visibleEdges.length < 28) {
      setBundledPaths([]);
      return;
    }

    const MAX_BUNDLE_EDGES = 200;
    const edgeList = visibleEdges.toArray() as unknown as cytoscape.EdgeSingular[];
    visibleEdges.addClass("bundled-edge");

    const rankedEdges = edgeList
      .map((edge) => {
        const confidence = Number(edge.data("confidence") ?? 0.65);
        const degreeScore = (edge.source().degree(false) + edge.target().degree(false)) * 0.025;
        return { edge, score: confidence * 0.72 + degreeScore };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_BUNDLE_EDGES)
      .map((item) => item.edge);

    if (rankedEdges.length < 18) {
      setBundledPaths([]);
      return;
    }

    type Vec = { x: number; y: number };
    type BundleEdge = {
      id: string;
      sourceId: string;
      targetId: string;
      source: Vec;
      target: Vec;
      points: Vec[];
    };

    const subdivisions = 7;
    const makeLinearPoints = (a: Vec, b: Vec) =>
      Array.from({ length: subdivisions + 1 }, (_, idx) => {
        const tRatio = idx / subdivisions;
        return {
          x: a.x + (b.x - a.x) * tRatio,
          y: a.y + (b.y - a.y) * tRatio
        };
      });

    const length = (a: Vec, b: Vec) => Math.hypot(b.x - a.x, b.y - a.y);
    const dot = (a: Vec, b: Vec) => a.x * b.x + a.y * b.y;
    const midpoint = (a: Vec, b: Vec) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
    const sub = (a: Vec, b: Vec) => ({ x: a.x - b.x, y: a.y - b.y });

    const bundleEdges: BundleEdge[] = rankedEdges.map((edge) => {
      const source = edge.source().renderedPosition();
      const target = edge.target().renderedPosition();
      return {
        id: edge.id(),
        sourceId: edge.source().id(),
        targetId: edge.target().id(),
        source,
        target,
        points: makeLinearPoints(source, target)
      };
    });

    const compat: Array<Array<{ idx: number; w: number }>> = Array.from(
      { length: bundleEdges.length },
      () => []
    );
    const compatibilityThreshold = 0.22;
    for (let i = 0; i < bundleEdges.length; i += 1) {
      for (let j = i + 1; j < bundleEdges.length; j += 1) {
        const e = bundleEdges[i];
        const f = bundleEdges[j];
        const eVec = sub(e.target, e.source);
        const fVec = sub(f.target, f.source);
        const eLen = Math.max(1, Math.hypot(eVec.x, eVec.y));
        const fLen = Math.max(1, Math.hypot(fVec.x, fVec.y));
        const angleComp = Math.abs(dot(eVec, fVec) / (eLen * fLen));
        const scaleComp = 2 / (eLen / fLen + fLen / eLen);
        const midE = midpoint(e.source, e.target);
        const midF = midpoint(f.source, f.target);
        const midDist = length(midE, midF);
        const posComp = ((eLen + fLen) / 2) / (((eLen + fLen) / 2) + midDist);
        const visComp = Math.max(0, 1 - midDist / (eLen + fLen));
        const compatibility = angleComp * scaleComp * posComp * visComp;
        if (compatibility >= compatibilityThreshold) {
          compat[i].push({ idx: j, w: compatibility });
          compat[j].push({ idx: i, w: compatibility });
        }
      }
    }

    let step = 0.22;
    const iterations = bundleEdges.length > 140 ? 9 : 13;
    for (let iter = 0; iter < iterations; iter += 1) {
      const snapshot = bundleEdges.map((edge) => edge.points.map((p) => ({ ...p })));
      for (let edgeIdx = 0; edgeIdx < bundleEdges.length; edgeIdx += 1) {
        const edge = bundleEdges[edgeIdx];
        const links = compat[edgeIdx];
        if (!links.length) continue;
        for (let pointIdx = 1; pointIdx < subdivisions; pointIdx += 1) {
          const point = snapshot[edgeIdx][pointIdx];
          const prev = snapshot[edgeIdx][pointIdx - 1];
          const next = snapshot[edgeIdx][pointIdx + 1];
          const spring = {
            x: (prev.x + next.x) * 0.5 - point.x,
            y: (prev.y + next.y) * 0.5 - point.y
          };
          let attractX = 0;
          let attractY = 0;
          let totalW = 0;
          for (const link of links) {
            const peerPoint = snapshot[link.idx][pointIdx];
            const dx = peerPoint.x - point.x;
            const dy = peerPoint.y - point.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 260) continue;
            const influence = link.w / (1 + dist * 0.02);
            attractX += dx * influence;
            attractY += dy * influence;
            totalW += influence;
          }
          const bundleForce =
            totalW > 0
              ? {
                  x: attractX / totalW,
                  y: attractY / totalW
                }
              : { x: 0, y: 0 };
          edge.points[pointIdx] = {
            x: point.x + (spring.x * 0.34 + bundleForce.x * 0.66) * step,
            y: point.y + (spring.y * 0.34 + bundleForce.y * 0.66) * step
          };
        }
      }
      step *= 0.88;
    }

    const toSmoothPath = (points: Vec[]) => {
      if (points.length < 2) return "";
      if (points.length === 2) {
        return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
      }
      let d = `M ${points[0].x} ${points[0].y}`;
      for (let idx = 1; idx < points.length - 1; idx += 1) {
        const current = points[idx];
        const next = points[idx + 1];
        const mid = { x: (current.x + next.x) / 2, y: (current.y + next.y) / 2 };
        d += ` Q ${current.x} ${current.y} ${mid.x} ${mid.y}`;
      }
      const last = points[points.length - 1];
      d += ` T ${last.x} ${last.y}`;
      return d;
    };

    const paths = bundleEdges.map((edge, idx) => {
      const sourceCluster = communityByNodeRef.current[edge.sourceId] || "";
      const stroke = communityMetaRef.current[sourceCluster]?.stroke || "rgba(79,70,229,0.34)";
      const degreeWeight = Math.max(0, compat[idx].length);
      const center = edge.points[Math.floor(edge.points.length / 2)];
      return {
        id: edge.id,
        path: toSmoothPath(edge.points),
        stroke,
        width: Math.max(0.9, Math.min(3.2, 0.95 + Math.log(degreeWeight + 1) * 0.72)),
        opacity: Math.min(0.28, 0.08 + Math.log(degreeWeight + 2) * 0.06),
        count: degreeWeight,
        labelX: center.x,
        labelY: center.y
      };
    });

    setBundledPaths(paths);
  };

  const scheduleBundleRefresh = () => {
    if (bundleRafRef.current !== null) return;
    bundleRafRef.current = window.requestAnimationFrame(() => {
      bundleRafRef.current = null;
      refreshEdgeBundles();
    });
  };

  const applyLayout = (layoutType: "force" | "hierarchy" | "timeline") => {
    const cy = cyRef.current;
    if (!cy) return;
    if (layoutType === "force") {
      setTimelineGuides([]);
      cy.layout({
        name: "cose",
        animate: true,
        fit: true,
        padding: 100,
        randomize: false,
        nodeRepulsion: () => 450000,
        idealEdgeLength: () => 280,
        edgeElasticity: () => 32,
        gravity: 0.1,
        numIter: 2500,
        nestingFactor: 0.1,
        gravityRangeCompound: 1.5,
        gravityCompound: 1.0,
        gravityRange: 3.8
      } as any).run();
      return;
    }
    if (layoutType === "hierarchy") {
      setTimelineGuides([]);
      cy.layout({ name: "breadthfirst", directed: true, padding: 110, spacingFactor: 1.35 }).run();
      return;
    }
    const nodes = cy.nodes();
    const years = nodes
      .map((n: cytoscape.NodeSingular) => n.data("full")?.year)
      .filter(Boolean) as number[];
    const uniqueYears = Array.from(new Set(years)).sort((a, b) => a - b);
    const minYear = uniqueYears[0] ?? 2000;
    const maxYear = uniqueYears[uniqueYears.length - 1] ?? 2026;
    const width = containerRef.current?.clientWidth ?? 800;
    const height = containerRef.current?.clientHeight ?? 600;
    const leftPad = timelinePadding.left;
    const rightPad = timelinePadding.right;
    const topPad = timelinePadding.top;
    const bottomPad = timelinePadding.bottom;
    const span = Math.max(1, maxYear - minYear);
    const usableWidth = Math.max(60, width - leftPad - rightPad);
    const yearToX = new Map<number, number>();
    uniqueYears.forEach((year) => {
      yearToX.set(year, leftPad + ((year - minYear) / span) * usableWidth);
    });
    const positions: Record<string, { x: number; y: number }> = {};
    const groups: Record<number, string[]> = {};
    nodes.forEach((n: cytoscape.NodeSingular) => {
      const year = n.data("full")?.year ?? minYear;
      if (!groups[year]) groups[year] = [];
      groups[year].push(n.id());
    });
    Object.keys(groups).forEach((yearStr) => {
      const year = Number(yearStr);
      const list = groups[year]
        .slice()
        .sort((aId, bId) => {
          const aCommunity = communityByNodeRef.current[aId] || "";
          const bCommunity = communityByNodeRef.current[bId] || "";
          if (aCommunity !== bCommunity) return aCommunity.localeCompare(bCommunity);
          const aNode = cy.$id(aId).data("full");
          const bNode = cy.$id(bId).data("full");
          const aScore = Number(aNode?.citation_count ?? aNode?.pagerank ?? 0);
          const bScore = Number(bNode?.citation_count ?? bNode?.pagerank ?? 0);
          return bScore - aScore;
        });
      const x = yearToX.get(year) ?? leftPad + ((year - minYear) / span) * usableWidth;
      const usableHeight = Math.max(80, height - topPad - bottomPad);
      const gap = list.length > 1 ? Math.max(12, Math.min(38, usableHeight / (list.length - 1))) : 0;
      const totalSpan = (list.length - 1) * gap;
      const startY = topPad + Math.max(0, (usableHeight - totalSpan) / 2);
      list.forEach((id, idx) => {
        const y = Math.min(height - bottomPad, startY + idx * gap);
        positions[id] = { x, y };
      });
    });
    cy.layout({ name: "preset", positions, animate: true, fit: true, padding: 110 }).run();
  };

  useEffect(() => {
    fetchData(subField, yearRange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadSubfields = async (activeOnly = true) => {
    try {
      const data = await fetchSubfields(activeOnly);
      if (activeOnly) setSubfields(data);
      else setAllSubfields(data);
    } catch {
      // ignore
    }
  };

  const handleCreateSubfield = async () => {
    if (!newSubfieldName.trim()) return;
    try {
      await createSubfield({ name: newSubfieldName.trim(), active: 1 });
      setNewSubfieldName("");
      await loadSubfields(true);
      await loadSubfields(false);
      message.success(t("msg.subfield_added"));
    } catch {
      message.error(t("msg.subfield_add_failed"));
    }
  };

  const handleToggleSubfield = async (item: Subfield) => {
    try {
      await updateSubfield(item.id, { active: item.active === 1 ? 0 : 1 });
      await loadSubfields(true);
      await loadSubfields(false);
    } catch {
      message.error(t("msg.subfield_update_failed"));
    }
  };

  const handleDeleteSubfield = async (item: Subfield) => {
    try {
      await deleteSubfield(item.id);
      await loadSubfields(true);
      await loadSubfields(false);
    } catch {
      message.error(t("msg.subfield_delete_failed"));
    }
  };

  useEffect(() => {
    loadSubfields(true);
  }, []);

  const refreshSyncStatus = async () => {
    try {
      const data = await fetchSyncStatus();
      setSyncStatus(data);
    } catch {
      // ignore
    }
  };

  const refreshJobBoard = async () => {
    setJobLoading(true);
    try {
      const [jobs, alerts] = await Promise.all([fetchJobHistory({ limit: 80 }), fetchJobAlerts({ resolved: false, limit: 80 })]);
      setJobRuns(jobs.items);
      setJobAlerts(alerts.items);
    } catch {
      // ignore
    } finally {
      setJobLoading(false);
    }
  };

  const refreshCapsules = async () => {
    try {
      const [list, board] = await Promise.all([fetchIdeaCapsules({ limit: 200 }), fetchIdeaCapsuleBoard()]);
      setCapsules(list.items);
      setCapsuleBoard(board.board || {});
    } catch {
      // ignore
    }
  };

  const refreshChatSessions = async () => {
    try {
      const sessions = await fetchChatSessions(30);
      setChatSessions(sessions);
      if (!chatSessionId && sessions.length > 0) {
        setChatSessionId(sessions[0].id);
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    refreshSyncStatus();
    refreshJobBoard();
    refreshCapsules();
    refreshChatSessions();
    const id = window.setInterval(refreshSyncStatus, 10000);
    const jobId = window.setInterval(refreshJobBoard, 15000);
    return () => {
      window.clearInterval(id);
      window.clearInterval(jobId);
    };
  }, []);

  useEffect(() => {
    if (subfieldModalOpen) {
      loadSubfields(false);
    } else {
      loadSubfields(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subfieldModalOpen]);

  useEffect(() => {
    if (!chatSessionId) {
      setChatHistory([]);
      return;
    }
    fetchChatMessages(chatSessionId, 200)
      .then((rows) => setChatHistory(rows))
      .catch(() => {
        // ignore
      });
  }, [chatSessionId]);

  useEffect(() => {
    const hide = () => setMenu((prev) => ({ ...prev, visible: false }));
    const onResize = () => {
      scheduleHullRefresh();
      scheduleBundleRefresh();
    };
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const update = () => {
      scheduleHullRefresh();
      scheduleBundleRefresh();
    };
    cy.on("zoom pan layoutstop dragfree add remove", update);
    update();
    return () => {
      cy.off("zoom pan layoutstop dragfree add remove", update);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showClusterHull, edgeBundling, graph, layout, edgeFocus]);

  useEffect(() => {
    return () => {
      if (hullRafRef.current !== null) {
        window.cancelAnimationFrame(hullRafRef.current);
      }
      if (bundleRafRef.current !== null) {
        window.cancelAnimationFrame(bundleRafRef.current);
      }
    };
  }, []);

  const loadNodeWorkspace = async (paperId: number) => {
    try {
      const [noteData, taskData, experimentData, backlinkData, schemaData, metricData, provenanceData] = await Promise.all([
        fetchPaperNotes(paperId),
        fetchTasks({ paper_id: paperId }),
        fetchExperiments({ paper_id: paperId }),
        fetchPaperBacklinks(paperId),
        fetchPaperSchema(paperId),
        fetchMetricLeaderboard({ metric: metricType, limit: 20 }),
        fetchMetricProvenance({ paper_id: paperId, limit: 60 })
      ]);
      setNote(noteData);
      setTasks(taskData);
      setExperiments(experimentData);
      setBacklinks(backlinkData);
      setPaperSchema(schemaData);
      setMetricLeaderboard(metricData);
      setMetricProvenance(provenanceData);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (!selected) {
      setNote(null);
      setTasks([]);
      setExperiments([]);
      setBacklinks(null);
      setPaperSchema(null);
      setMetricProvenance(null);
      return;
    }
    loadNodeWorkspace(selected.id);
    if (focusMode) {
      loadNeighbors();
    }
  }, [selected?.id, focusMode, metricType]);

  useEffect(() => {
    if (workbenchTab !== "schema") return;
    fetchSchemaOntology(2)
      .then((data) => setSchemaOntology(data))
      .catch(() => {
        // ignore
      });
  }, [workbenchTab]);

  useEffect(() => {
    if (!containerRef.current) return;
    if (!cyRef.current) {
      cyRef.current = cytoscape({
        container: containerRef.current,
        elements: [],
        style: [
          {
            selector: "node",
            style: {
              "background-color": "data(color)",
              label: "data(displayLabel)",
              color: "#0f172a",
              "font-size": 10,
              "text-wrap": "wrap",
              "text-max-width": "120",
              "min-zoomed-font-size": 5,
              width: "data(size)",
              height: "data(size)",
              "text-valign": "center",
              "text-halign": "center",
              "text-margin-y": 0,
              "text-background-color": "#ffffff",
              "text-background-opacity": (node: cytoscape.NodeSingular) =>
                Number(node.data("labelBgOpacity") ?? 0),
              "text-background-padding": "1.2",
              "text-opacity": (node: cytoscape.NodeSingular) => Number(node.data("labelOpacity") ?? 0),
              "background-opacity": 0.9,
              "border-width": 1,
              "border-color": "rgba(255,255,255,0.8)"
            }
          },
          {
            selector: "edge",
            style: {
              width: 1,
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "#94a3b8",
              "line-color": "#cbd5e1",
              opacity: 0.4
            }
          },
          {
            selector: "edge.bundled-edge",
            style: {
              width: 0.35,
              opacity: 0.03,
              "line-color": "#cbd5e1",
              "target-arrow-shape": "none"
            }
          },
          {
            selector: "edge.intent-build_on",
            style: {
              "line-color": "#16a34a",
              "target-arrow-color": "#16a34a"
            }
          },
          {
            selector: "edge.intent-contrast",
            style: {
              "line-color": "#ef4444",
              "target-arrow-color": "#ef4444"
            }
          },
          {
            selector: "edge.intent-use_as_baseline",
            style: {
              "line-color": "#f59e0b",
              "target-arrow-color": "#f59e0b"
            }
          },
          {
            selector: "edge.intent-mention",
            style: {
              "line-color": "#94a3b8",
              "target-arrow-color": "#94a3b8"
            }
          },
          {
            selector: ":selected",
            style: {
              "border-width": 2,
              "border-color": "#0ea5e9"
            }
          },
          {
            selector: ".highlight",
            style: {
              "border-width": 4,
              "border-color": "#f97316"
            }
          },
          {
            selector: ".highlight-edge",
            style: {
              width: 2,
              "line-color": "#f97316",
              "target-arrow-color": "#f97316"
            }
          },
          {
            selector: ".dim",
            style: {
              opacity: 0.2
            }
          },
          {
            selector: ".path-step",
            style: {
              "border-width": 6,
              "border-color": "#22c55e"
            }
          },
          {
            selector: ".path-edge",
            style: {
              width: 3,
              "line-color": "#22c55e",
              "target-arrow-color": "#22c55e"
            }
          },
          {
            selector: ".focus-dim",
            style: {
              opacity: 0.08,
              "text-opacity": 0.08
            }
          },
          {
            selector: ".focus-edge",
            style: {
              width: 3,
              "line-color": "#0ea5e9",
              "target-arrow-color": "#0ea5e9"
            }
          },
          {
            selector: ".focus-node",
            style: {
              "border-width": 4,
              "border-color": "#0ea5e9"
            }
          },
          {
            selector: ".declutter-edge",
            style: {
              opacity: 0.03,
              width: 0.35
            }
          },
          {
            selector: ".declutter-node",
            style: {
              opacity: 0.16,
              "text-opacity": 0
            }
          }
        ]
      });

      cyRef.current.on("tap", "node", (evt: cytoscape.EventObject) => {
        const data = evt.target.data();
        setSelected(data.full);
        setZoteroKey(data.full?.zotero_item_key || "");
        setMenu((prev) => ({ ...prev, visible: false }));
        hoveredNodeIdRef.current = evt.target.id();
        updateNodeLabels();
      });
      cyRef.current.on("cxttap", "node", (evt: cytoscape.EventObject) => {
        const data = evt.target.data();
        setSelected(data.full);
        setZoteroKey(data.full?.zotero_item_key || "");
        const original = evt.originalEvent as MouseEvent | undefined;
        setMenu({
          visible: true,
          x: original?.clientX ?? 0,
          y: original?.clientY ?? 0,
          node: data.full ?? null
        });
      });
      cyRef.current.on("mouseover", "node", (evt: cytoscape.EventObject) => {
        hoveredNodeIdRef.current = evt.target.id();
        updateNodeLabels();
      });
      cyRef.current.on("mouseout", "node", () => {
        hoveredNodeIdRef.current = null;
        updateNodeLabels();
      });
      cyRef.current.on("zoom", () => {
        updateNodeLabels();
        applySemanticDeclutter();
        scheduleBundleRefresh();
      });
      cyRef.current.on("tap", () => {
        setMenu((prev) => ({ ...prev, visible: false }));
      });
    }
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("focus-dim focus-node");
    cy.edges().removeClass("focus-dim focus-edge");
    if (edgeIntentFilter !== "all") {
      cy.edges().forEach((edge) => {
        const intent = edge.data("intent");
        if (intent !== edgeIntentFilter) {
          edge.addClass("focus-dim");
        }
      });
    }
    if (!selected || edgeFocus === "all") {
      if (focusMode && selected) {
        const nodeId = selected.id.toString();
        const node = cy.$id(nodeId);
        cy.nodes().addClass("focus-dim");
        cy.edges().addClass("focus-dim");
        node.removeClass("focus-dim").addClass("focus-node");
        const neighbors = node.neighborhood();
        neighbors.removeClass("focus-dim").addClass("focus-edge");
        neighbors.nodes().removeClass("focus-dim");
      }
      applySemanticDeclutter();
      return;
    }
    const nodeId = selected.id.toString();
    cy.nodes().addClass("focus-dim");
    cy.edges().addClass("focus-dim");
    const node = cy.$id(nodeId);
    node.removeClass("focus-dim").addClass("focus-node");
    const selector = edgeFocus === "out" ? `[source = \"${nodeId}\"]` : `[target = \"${nodeId}\"]`;
    const edges = cy.edges(selector);
    const filteredEdges =
      edgeIntentFilter === "all"
        ? edges
        : edges.filter((edge) => edge.data("intent") === edgeIntentFilter);
    filteredEdges.removeClass("focus-dim").addClass("focus-edge");
    filteredEdges.sources().removeClass("focus-dim");
    filteredEdges.targets().removeClass("focus-dim");
    applySemanticDeclutter();
  }, [edgeFocus, edgeIntentFilter, focusMode, selected, graph, smartDeclutter]);

  useEffect(() => {
    fetchData(subField, yearRange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sizeBy, edgeRecentYears, minIndegree, lazyMode, nodeLimit, sortBy, uploadedOnly, edgeMinConfidence]);

  useEffect(() => {
    if (!graph || !cyRef.current) return;
    const cy = cyRef.current;
    cy.elements().remove();

    const elements: cytoscape.ElementDefinition[] = [];
    const ranked = graph.nodes
      .map((node) => ({
        id: node.id.toString(),
        score: Number(node.citation_count ?? node.pagerank ?? node.citation_velocity ?? 0)
      }))
      .sort((a, b) => b.score - a.score);
    const topRatio = ranked.length > 420 ? 0.04 : ranked.length > 220 ? 0.06 : ranked.length > 120 ? 0.08 : 0.1;
    const topCount = Math.max(1, Math.ceil(ranked.length * topRatio));
    const topLabelIds = new Set(ranked.slice(0, topCount).map((item) => item.id));
    graph.nodes.forEach((node) => {
      baseColors.current[node.id.toString()] = node.color;
      const compactLabel = shortLabel(node.label);
      elements.push({
        data: {
          id: node.id.toString(),
          label: compactLabel,
          displayLabel: "",
          fullLabel: compactLabel,
          labelOpacity: 0,
          labelBgOpacity: 0,
          topLabel: topLabelIds.has(node.id.toString()) ? 1 : 0,
          size: node.size,
          color: node.color,
          full: node
        }
      });
    });
    graph.edges.forEach((edge) => {
      elements.push({
        data: {
          id: edge.id,
          source: edge.source.toString(),
          target: edge.target.toString(),
          confidence: edge.confidence ?? 0.75,
          edge_source: edge.edge_source ?? null,
          intent: edge.intent ?? null,
          intent_confidence: edge.intent_confidence ?? null
        }
      });
    });
    cy.add(elements);
    cy.edges().forEach((edge) => {
      const intent = edge.data("intent");
      if (intent) {
        edge.addClass(`intent-${intent}`);
      }
    });
    detectCommunities();
    updateNodeLabels();
    applyLayout(layout);
    applySemanticDeclutter();
    scheduleHullRefresh();
    scheduleBundleRefresh();
    clearPathAnimation();
  }, [graph, showLabels, smartDeclutter, layout]);

  useEffect(() => {
    applyLayout(layout);
    scheduleHullRefresh();
    scheduleBundleRefresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

  useEffect(() => {
    if (!graph || !cyRef.current) return;
    const cy = cyRef.current;
    if (!strategy) {
      cy.nodes().removeClass("highlight dim");
      cy.edges().removeClass("highlight-edge");
      cy.nodes().forEach((n: cytoscape.NodeSingular) => {
        const base = baseColors.current[n.id()];
        if (base) n.data("color", base);
      });
      setHighlightOrder([]);
      clearPathAnimation();
      return;
    }
    const params: { strategy: "foundation" | "sota" | "cluster"; sub_field?: string; year_from?: number; year_to?: number } =
      { strategy };
    if (subField) params.sub_field = subField;
    if (yearRange) {
      params.year_from = yearRange[0];
      params.year_to = yearRange[1];
    }

    fetchRecommendations(params).then((data) => {
      cy.nodes().removeClass("highlight dim");
      cy.edges().removeClass("highlight-edge");

      if (data.strategy === "cluster" && data.clusters) {
        const palette = ["#f59e0b", "#10b981", "#3b82f6", "#a855f7", "#ef4444", "#14b8a6"];
        const colorMap: Record<string, string> = {};
        let idx = 0;
        Object.values(data.clusters).forEach((label) => {
          if (!colorMap[label]) {
            colorMap[label] = palette[idx % palette.length];
            idx += 1;
          }
        });
        cy.nodes().forEach((n: cytoscape.NodeSingular) => {
          const label = data.clusters?.[n.id()];
          const color = label ? colorMap[label] : baseColors.current[n.id()];
          n.data("color", color || baseColors.current[n.id()]);
        });
        setHighlightOrder([]);
        return;
      }

      // restore base colors for non-cluster strategies
      cy.nodes().forEach((n: cytoscape.NodeSingular) => {
        const base = baseColors.current[n.id()];
        if (base) n.data("color", base);
      });

      const ordered = (data.highlight_nodes || []).map((id) => id.toString());
      setHighlightOrder(ordered);
      const highlights = new Set(ordered);
      if (highlights.size === 0) return;
      cy.nodes().forEach((n: cytoscape.NodeSingular) => {
        if (highlights.has(n.id())) n.addClass("highlight");
        else n.addClass("dim");
      });
      cy.edges().forEach((e: cytoscape.EdgeSingular) => {
        if (highlights.has(e.source().id()) && highlights.has(e.target().id())) {
          e.addClass("highlight-edge");
        }
      });
    });
  }, [strategy, graph, subField, yearRange]);

  const clearPathAnimation = () => {
    if (!cyRef.current) return;
    animationRef.current.forEach((id) => window.clearTimeout(id));
    animationRef.current = [];
    cyRef.current.nodes().removeClass("path-step");
    cyRef.current.edges().removeClass("path-edge");
    setIsAnimating(false);
  };

  const playPathAnimation = () => {
    if (!cyRef.current || highlightOrder.length === 0) return;
    clearPathAnimation();
    setIsAnimating(true);
    const cy = cyRef.current;
    highlightOrder.forEach((nodeId, idx) => {
      const timeoutId = window.setTimeout(() => {
        const node = cy.$id(nodeId);
        node.addClass("path-step");
        if (idx > 0) {
          const prev = highlightOrder[idx - 1];
          const edge = cy.edges(
            `[source = \"${prev}\"][target = \"${nodeId}\"]`
          );
          edge.addClass("path-edge");
        }
        if (idx === highlightOrder.length - 1) {
          setIsAnimating(false);
        }
      }, idx * 700);
      animationRef.current.push(timeoutId);
    });
  };

  const subFieldOptions = useMemo(() => {
    if (subfields.length > 0) {
      return subfields.map((s) => ({ label: s.name, value: s.name }));
    }
    if (!graph) return [];
    return graph.meta.sub_fields.map((s) => ({ label: s, value: s }));
  }, [graph, subfields]);

  const pathNodeOptions = useMemo(() => {
    if (!graph?.nodes?.length) return [];
    return graph.nodes
      .slice()
      .sort((a, b) => (b.year || 0) - (a.year || 0))
      .slice(0, 400)
      .map((node) => ({
        label: `#${node.id} ${(node.label || "").replace(/\s+/g, " ").trim().slice(0, 42)}`,
        value: node.id
      }));
  }, [graph]);

  const shortLabel = (value: string) => {
    if (!value) return "";
    const compact = value.replace(/\s+/g, " ").trim();
    if (compact.length <= 36) return compact;
    return `${compact.slice(0, 33)}...`;
  };

  const updateNodeLabels = () => {
    const cy = cyRef.current;
    if (!cy) return;
    const zoom = cy.zoom();
    const total = cy.nodes().length;
    const labelsEnabled = showLabelsRef.current;
    const mode = labelModeRef.current;
    const selectedId = selectedNodeIdRef.current;
    cy.nodes().forEach((node: cytoscape.NodeSingular) => {
      const id = node.id();
      const base = node.data("fullLabel") || shortLabel(node.data("full")?.label || "");
      const isSelected = selectedId === id;
      const isHovered = hoveredNodeIdRef.current === id;
      const isTopLabel = Number(node.data("topLabel")) === 1;
      let shouldShow = false;
      if (!labelsEnabled) {
        shouldShow = isSelected || isHovered;
      } else if (mode === "all") {
        shouldShow = total <= 120 || zoom >= 1.15 || isSelected || isHovered;
      } else if (mode === "selected") {
        shouldShow = isSelected || isHovered;
      } else {
        // auto
        shouldShow =
          isSelected ||
          isHovered ||
          (isTopLabel && zoom >= (total > 220 ? 1.02 : 0.92)) ||
          (total <= 42 && zoom >= 0.8) ||
          (total <= 90 && zoom >= 1.0) ||
          zoom >= 1.42;
      }
      node.data("displayLabel", shouldShow ? base : "");
      node.data("labelOpacity", shouldShow ? 0.95 : 0);
      node.data("labelBgOpacity", shouldShow ? 0.52 : 0);
    });
  };

  const applySemanticDeclutter = () => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.edges().removeClass("declutter-edge");
    cy.nodes().removeClass("declutter-node");
    if (!smartDeclutter) {
      scheduleBundleRefresh();
      return;
    }

    const zoom = cy.zoom();
    const selectedId = selected ? selected.id.toString() : null;
    const nodeCount = cy.nodes().length;
    const edgeCount = cy.edges().length;

    if (edgeCount > 220 && zoom < 0.95) {
      cy.edges().forEach((edge) => {
        if (selectedId && (edge.source().id() === selectedId || edge.target().id() === selectedId)) return;
        edge.addClass("declutter-edge");
      });
    }

    if (nodeCount > 280 && zoom < 0.78) {
      cy.nodes().forEach((node) => {
        if (selectedId && node.id() === selectedId) return;
        if (node.degree(false) <= 1) {
          node.addClass("declutter-node");
        }
      });
    }
    scheduleBundleRefresh();
  };

  useEffect(() => {
    showLabelsRef.current = showLabels;
    labelModeRef.current = labelMode;
    selectedNodeIdRef.current = selected ? selected.id.toString() : null;
    updateNodeLabels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showLabels, labelMode, selected, graph]);

  const timelineData = useMemo(() => {
    if (!graph) return [];
    return graph.meta.year_counts_all ?? graph.meta.year_counts ?? [];
  }, [graph]);

  const timelineMin = graph?.meta.year_min_all ?? graph?.meta.year_min ?? 2000;
  const timelineMax = graph?.meta.year_max_all ?? graph?.meta.year_max ?? 2026;
  const displayRange = isDragging && dragRange ? dragRange : pendingRange ?? yearRange;
  const bubbleLabel =
    displayRange && (isDragging || pendingRange)
      ? `${displayRange[0]}–${displayRange[1]}`
      : null;

  const handleFilterChange = (nextSub?: string, nextRange?: [number, number] | null) => {
    setSubField(nextSub);
    if (nextRange) setYearRange(nextRange);
    setPendingRange(null);
    fetchData(nextSub, nextRange ?? yearRange);
  };

  const resetYearFilter = () => {
    setPendingRange(null);
    handleFilterChange(subField, [timelineMin, timelineMax]);
  };

  const loadNeighbors = async () => {
    if (!selected || !cyRef.current) return;
    try {
      const data = await fetchNeighbors({ node_id: selected.id, depth: 1, limit: 200 });
      const cy = cyRef.current;
      data.nodes.forEach((node) => {
        baseColors.current[node.id.toString()] = node.color;
        if (cy.$id(node.id.toString()).empty()) {
          cy.add({
            data: {
              id: node.id.toString(),
              label: shortLabel(node.label),
              displayLabel: "",
              fullLabel: shortLabel(node.label),
              size: node.size ?? 24,
              color: node.color,
              full: node
            }
          });
        }
      });
      data.edges.forEach((edge) => {
        if (cy.$id(edge.id).empty()) {
          cy.add({
            data: {
              id: edge.id,
              source: edge.source.toString(),
              target: edge.target.toString(),
              confidence: edge.confidence ?? 0.75,
              edge_source: edge.edge_source ?? null,
              intent: edge.intent ?? null,
              intent_confidence: edge.intent_confidence ?? null
            }
          });
          if (edge.intent) {
            cy.$id(edge.id).addClass(`intent-${edge.intent}`);
          }
        }
      });
      detectCommunities();
      applyLayout(layout);
      applySemanticDeclutter();
      scheduleHullRefresh();
      scheduleBundleRefresh();
    } catch {
      message.error(t("msg.neighbors_failed"));
    }
  };

  const refreshTimelinePositions = () => {
    if (!timelineRef.current) return;
    const items = Array.from(
      timelineRef.current.querySelectorAll<HTMLElement>("[data-year]")
    );
    timelinePositions.current = items.map((el) => {
      const rect = el.getBoundingClientRect();
      return { year: Number(el.dataset.year), x: rect.left + rect.width / 2 };
    });
  };

  useEffect(() => {
    refreshTimelinePositions();
    const onResize = () => refreshTimelinePositions();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [timelineData]);

  const nearestYear = (clientX: number) => {
    const positions = timelinePositions.current;
    if (!positions.length) return null;
    let best = positions[0];
    let min = Math.abs(clientX - best.x);
    for (const pos of positions) {
      const dist = Math.abs(clientX - pos.x);
      if (dist < min) {
        best = pos;
        min = dist;
      }
    }
    return best.year;
  };

  const handleTimelineMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    dragStartX.current = e.clientX;
    dragAnchorYear.current = null;
    setIsDragging(false);
    setDragRange(null);
  };

  const handleTimelineMouseMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (dragStartX.current === null) return;
    const delta = Math.abs(e.clientX - dragStartX.current);
    if (!isDragging && delta < 4) return;
    if (!isDragging) {
      setIsDragging(true);
      const anchor = nearestYear(dragStartX.current);
      dragAnchorYear.current = anchor;
      if (anchor !== null) setDragRange([anchor, anchor]);
    }
    const anchor = dragAnchorYear.current;
    const current = nearestYear(e.clientX);
    if (anchor !== null && current !== null) {
      setDragRange([Math.min(anchor, current), Math.max(anchor, current)]);
    }
  };

  const handleTimelineMouseUp = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (dragStartX.current === null) return;
    const wasDragging = isDragging;
    const current = nearestYear(e.clientX);
    dragStartX.current = null;
    dragAnchorYear.current = null;
    if (wasDragging && dragRange) {
      setPendingRange(dragRange);
    } else if (current !== null) {
      setPendingRange([current, current]);
    }
    setIsDragging(false);
    setDragRange(null);
  };

  const applyPendingRange = () => {
    if (!pendingRange) return;
    handleFilterChange(subField, pendingRange);
  };

  const cancelPendingRange = () => {
    setPendingRange(null);
  };

  useEffect(() => {
    runComparison();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareQueue.join(",")]);

  const maybeAutoPush = async (paperId: number, zoteroKey?: string | null) => {
    if (!autoPush || !zoteroKey) return;
    try {
      await pushZotero(paperId);
      message.success(t("zotero.auto_push_ok"));
    } catch {
      message.error(t("zotero.auto_push_fail"));
    }
  };

  const addToComparison = (paperId: number) => {
    setCompareQueue((prev) => {
      if (prev.includes(paperId)) return prev;
      return [...prev, paperId].slice(-8);
    });
  };

  const removeFromComparison = (paperId: number) => {
    setCompareQueue((prev) => prev.filter((id) => id !== paperId));
  };

  const runComparison = async () => {
    if (compareQueue.length === 0) {
      setCompareData(null);
      return;
    }
    setCompareLoading(true);
    try {
      const data = await comparePapers({ paper_ids: compareQueue });
      setCompareData(data);
    } finally {
      setCompareLoading(false);
    }
  };

  const classifyIntents = async () => {
    const cy = cyRef.current;
    const edges = cy?.edges().map((edge) => edge.id()) ?? [];
    if (!edges.length) {
      message.warning(t("details.select_hint"));
      return;
    }
    const res = await classifyCitationIntent({ edge_ids: edges, limit: 200 });
    message.success(`${t("sync.processed")} ${res.updated}`);
    await fetchData(subField, yearRange);
  };

  const highlightShortestPath = async () => {
    if (!pathSourceId || !pathTargetId || !cyRef.current) {
      message.warning(t("graph.path_missing"));
      return;
    }
    setPathLoading(true);
    try {
      const data = await fetchShortestPath({
        source_id: pathSourceId,
        target_id: pathTargetId,
        direction: edgeFocus === "all" ? "any" : edgeFocus
      });
      const cy = cyRef.current;
      if (!cy) return;
      cy.nodes().removeClass("path-step");
      cy.edges().removeClass("path-edge");
      if (!data.path.nodes.length) {
        message.warning(t("graph.path_not_found"));
        return;
      }
      data.path.nodes.forEach((id) => cy.$id(String(id)).addClass("path-step"));
      data.path.edges.forEach(([source, target]) => {
        const forward = cy.edges(`[source = \"${source}\"][target = \"${target}\"]`);
        if (forward.nonempty()) {
          forward.addClass("path-edge");
          return;
        }
        cy.edges(`[source = \"${target}\"][target = \"${source}\"]`).addClass("path-edge");
      });
      message.success(`${t("graph.path_len")}: ${data.path.distance ?? 0}`);
    } finally {
      setPathLoading(false);
    }
  };

  const askPaperQa = async () => {
    const query = qaQuery.trim();
    if (!query) return;
    const paperIds =
      qaMode === "single"
        ? selected
          ? [selected.id]
          : []
        : compareQueue;
    if (!paperIds.length) {
      message.warning(t("qa.select_papers"));
      return;
    }
    setQaLoading(true);
    try {
      setQaResult({
        query,
        paper_ids: paperIds,
        answer: "",
        sources: [],
        context_count: 0
      });
      let activeSessionId = chatSessionId;
      if (!activeSessionId) {
        const session = await createChatSession({ title: query.slice(0, 40), language: "zh" });
        activeSessionId = session.id;
        setChatSessionId(session.id);
        await refreshChatSessions();
      }
      await streamChatWithPapers(
        {
          query,
          paper_ids: paperIds,
          top_k: 8,
          language: navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en",
          session_id: activeSessionId,
          use_memory: true
        },
        {
          onMeta: (meta) => {
            if (meta.session_id) {
              setChatSessionId(meta.session_id);
            }
            setQaResult((prev) => ({
              query: meta.query,
              paper_ids: meta.paper_ids,
              answer: prev?.answer ?? "",
              sources: prev?.sources ?? [],
              context_count: meta.context_count,
              session_id: meta.session_id,
              routes: meta.routes
            }));
          },
          onDelta: (delta) =>
            setQaResult((prev) => ({
              query,
              paper_ids: paperIds,
              answer: `${prev?.answer || ""}${delta}`,
              sources: prev?.sources || [],
              context_count: prev?.context_count || 0
            })),
          onSources: (sources, traceScore) =>
            setQaResult((prev) => ({
              query,
              paper_ids: paperIds,
              answer: prev?.answer || "",
              sources,
              context_count: prev?.context_count || 0,
              trace_score: traceScore
            })),
          onDone: async () => {
            if (activeSessionId) {
              const rows = await fetchChatMessages(activeSessionId, 120);
              setChatHistory(rows);
            }
          }
        }
      );
    } catch {
      // Fallback to non-streaming API when SSE is unavailable.
      const result = await chatWithPapers({
        query,
        paper_ids: paperIds,
        top_k: 8,
        language: navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en",
        session_id: chatSessionId,
        use_memory: true
      });
      setQaResult(result);
      if (result.session_id) {
        setChatSessionId(result.session_id);
        const rows = await fetchChatMessages(result.session_id, 120);
        setChatHistory(rows);
      }
    } finally {
      setQaLoading(false);
    }
  };

  const buildPathData = async () => {
    if (!strategy || (strategy !== "foundation" && strategy !== "sota")) {
      message.warning(t("msg.select_strategy"));
      return null;
    }
    const params: { strategy: "foundation" | "sota"; sub_field?: string; year_from?: number; year_to?: number } =
      { strategy };
    if (subField) params.sub_field = subField;
    if (yearRange) {
      params.year_from = yearRange[0];
      params.year_to = yearRange[1];
    }
    const data = await exportPath(params);
    return data;
  };

  const handleExportCsv = async () => {
    const data = await buildPathData();
    if (!data) return;
    const rows = data.path.map((p, idx) => ({
      order: idx + 1,
      title: p.title ?? "",
      authors: p.authors ?? "",
      year: p.year ?? "",
      venue: p.venue ?? "",
      doi: p.doi ?? "",
      url: p.url ?? ""
    }));
    const header = ["order", "title", "authors", "year", "venue", "doi", "url"];
    const escape = (value: string | number) => {
      const str = String(value ?? "");
      const escaped = str.replace(/\"/g, "\"\"");
      return `"${escaped}"`;
    };
    const csv = [
      header.join(","),
      ...rows.map((r) => header.map((h) => escape(r[h as keyof typeof r] as string | number)).join(","))
    ].join("\\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `reading_path_${data.strategy}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportMarkdown = async () => {
    const data = await buildPathData();
    if (!data) return;
    const lines = [
      `# Reading Path (${data.strategy})`,
      "",
      ...data.path.map(
        (p, idx) =>
          `${idx + 1}. **${p.title ?? ""}** (${p.year ?? ""})  \\\n` +
          `${p.authors ?? ""}  \\\n` +
          `${p.venue ?? ""}  \\\n` +
          `${p.doi ? `DOI: ${p.doi}` : ""}  \\\n` +
          `${p.url ?? ""}`
      )
    ];
    const blob = new Blob([lines.join("\\n")], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `reading_path_${data.strategy}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportPdf = async () => {
    const data = await buildPathData();
    if (!data) return;
    const doc = new jsPDF({ orientation: "p", unit: "pt", format: "a4" });
    let y = 40;
    doc.setFontSize(16);
    doc.text(`Reading Path (${data.strategy})`, 40, y);
    y += 24;
    doc.setFontSize(10);
    data.path.forEach((p, idx) => {
      const lines = [
        `${idx + 1}. ${p.title ?? ""} (${p.year ?? ""})`,
        `${p.authors ?? ""}`,
        `${p.venue ?? ""}`,
        `${p.doi ? `DOI: ${p.doi}` : ""}`,
        `${p.url ?? ""}`
      ].filter(Boolean);
      lines.forEach((line) => {
        const wrapped = doc.splitTextToSize(line, 520);
        wrapped.forEach((segment: string) => {
          if (y > 780) {
            doc.addPage();
            y = 40;
          }
          doc.text(segment, 40, y);
          y += 14;
        });
      });
      y += 8;
    });
    doc.save(`reading_path_${data.strategy}.pdf`);
  };

  const handleExportBibtex = async () => {
    const data = await buildPathData();
    if (!data) return;
    const usedKeys = new Map<string, number>();
    const slug = (value: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, "");
    const makeKey = (p: { authors?: string | null; year?: number | null; title?: string | null }, idx: number) => {
      const author = p.authors?.split(",")[0] ?? "paper";
      const last = author.trim().split(" ").pop() ?? "paper";
      const year = p.year ?? "nd";
      const titleWord = (p.title ?? "paper").split(/\s+/)[0] ?? "paper";
      let key = slug(`${last}${year}${titleWord}`);
      if (!key) key = `paper${idx + 1}`;
      const count = usedKeys.get(key) ?? 0;
      usedKeys.set(key, count + 1);
      if (count > 0) key = `${key}${count + 1}`;
      return key;
    };

    const entries = data.path.map((p, idx) => {
      const key = makeKey(p, idx);
      const venue = p.venue ?? "";
      const isJournal = /journal|transactions|letters|magazine/i.test(venue);
      const isConference = /conference|symposium|workshop|proceedings|meeting/i.test(venue);
      const hasVenue = Boolean(venue);
      const hasYear = Boolean(p.year);
      const hasUrl = Boolean(p.url);
      let entryType: "article" | "inproceedings" | "misc" = "inproceedings";
      if (isJournal && !isConference) entryType = "article";
      if (!hasVenue && (hasUrl || p.doi)) entryType = "misc";
      const venueField =
        hasVenue && entryType === "article"
          ? `  journal = {${venue}}`
          : hasVenue && entryType === "inproceedings"
            ? `  booktitle = {${venue}}`
            : "";
      const fields = [
        `  title = {${p.title ?? ""}}`,
        `  author = {${p.authors ?? ""}}`,
        hasYear ? `  year = {${p.year ?? ""}}` : "",
        venueField,
        p.doi ? `  doi = {${p.doi}}` : "",
        p.url ? `  url = {${p.url}}` : "",
        entryType === "misc" && hasUrl ? `  howpublished = {\\url{${p.url}}}` : ""
      ].filter(Boolean);
      return `@${entryType}{${key},\n${fields.join(",\n")}\n}`;
    });

    const blob = new Blob([entries.join("\n\n")], { type: "application/x-bibtex;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `reading_path_${data.strategy}.bib`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const filterPanel = (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <div>
        <Text type="secondary">{t("filters.subfield")}</Text>
        <Select
          allowClear
          style={{ width: "100%", marginTop: 8 }}
          options={subFieldOptions}
          value={subField}
          onChange={(value) => handleFilterChange(value, yearRange)}
          placeholder={t("filters.subfield")}
        />
        <Button type="link" onClick={() => setSubfieldModalOpen(true)}>
          {t("filters.manage_subfields")}
        </Button>
      </div>
      <div>
        <Text type="secondary">{t("filters.uploaded_only")}</Text>
        <Space style={{ marginTop: 8 }}>
          <Switch checked={uploadedOnly} onChange={setUploadedOnly} />
        </Space>
      </div>
      <div>
        <Text type="secondary">{t("filters.reco")}</Text>
        <Select
          allowClear
          style={{ width: "100%", marginTop: 8 }}
          value={strategy}
          onChange={(value) => setStrategy(value)}
          options={[
            { label: t("reco.foundation"), value: "foundation" },
            { label: t("reco.sota"), value: "sota" },
            { label: t("reco.cluster"), value: "cluster" }
          ]}
          placeholder={t("filters.reco")}
        />
      </div>
      <div>
        <Text type="secondary">{t("filters.layout")}</Text>
        <Select
          style={{ width: "100%", marginTop: 8 }}
          value={layout}
          onChange={(value) => setLayout(value as "force" | "hierarchy" | "timeline")}
          options={[
            { label: t("layout.force"), value: "force" },
            { label: t("layout.hierarchy"), value: "hierarchy" },
            { label: t("layout.timeline"), value: "timeline" }
          ]}
        />
      </div>
      <div>
        <Text type="secondary">{t("filters.node_size")}</Text>
        <Select
          style={{ width: "100%", marginTop: 8 }}
          value={sizeBy}
          onChange={(value) => setSizeBy(value as "citations" | "pagerank")}
          options={[
            { label: t("size.citations"), value: "citations" },
            { label: t("size.pagerank"), value: "pagerank" }
          ]}
        />
      </div>
      <div>
        <Text type="secondary">{t("filters.edge_filter")}</Text>
        <Select
          allowClear
          style={{ width: "100%", marginTop: 8 }}
          value={edgeRecentYears}
          onChange={(value) => setEdgeRecentYears(value ?? undefined)}
          options={[
            { label: t("edge.last3"), value: 3 },
            { label: t("edge.last5"), value: 5 }
          ]}
        />
      </div>
      <div>
        <Text type="secondary">{t("filters.edge_confidence")}</Text>
        <Slider
          min={0}
          max={1}
          step={0.05}
          value={edgeMinConfidence}
          onChange={(value) => setEdgeMinConfidence(value as number)}
          tooltip={{ formatter: (v) => `${Math.round((v || 0) * 100)}%` }}
        />
      </div>
      <div>
        <Text type="secondary">{t("filters.min_indegree")}</Text>
        <InputNumber
          style={{ width: "100%", marginTop: 8 }}
          min={0}
          value={minIndegree}
          onChange={(value) => setMinIndegree(value ?? undefined)}
        />
      </div>
      <div>
        <Text type="secondary">{t("filters.lazy")}</Text>
        <Space style={{ marginTop: 8 }}>
          <Switch checked={lazyMode} onChange={setLazyMode} />
          <InputNumber
            min={50}
            step={50}
            value={nodeLimit}
            onChange={(value) => setNodeLimit(value ?? 200)}
            disabled={!lazyMode}
          />
          <Text type="secondary">{t("filters.nodes")}</Text>
        </Space>
        {lazyMode && (
          <Button
            style={{ marginTop: 8 }}
            onClick={() => setNodeLimit((prev) => prev + 100)}
          >
            {t("filters.load_more")}
          </Button>
        )}
      </div>
      <div>
        <Text type="secondary">{t("filters.sort_nodes")}</Text>
        <Select
          style={{ width: "100%", marginTop: 8 }}
          value={sortBy}
          onChange={(value) => setSortBy(value as "citation_count" | "year" | "pagerank")}
          options={[
            { label: t("sort.citations"), value: "citation_count" },
            { label: t("sort.year"), value: "year" },
            { label: t("sort.pagerank"), value: "pagerank" }
          ]}
        />
      </div>
      <Space size="small" wrap>
        <Button onClick={handleExportCsv} disabled={!strategy || strategy === "cluster"}>
          {t("export.csv")}
        </Button>
        <Button onClick={handleExportMarkdown} disabled={!strategy || strategy === "cluster"}>
          {t("export.md")}
        </Button>
        <Button onClick={handleExportPdf} disabled={!strategy || strategy === "cluster"}>
          {t("export.pdf")}
        </Button>
        <Button onClick={handleExportBibtex} disabled={!strategy || strategy === "cluster"}>
          {t("export.bib")}
        </Button>
        <Button
          onClick={playPathAnimation}
          disabled={highlightOrder.length === 0 || isAnimating}
        >
          {t("path.play")}
        </Button>
        <Button onClick={clearPathAnimation} disabled={!isAnimating && highlightOrder.length === 0}>
          {t("path.clear")}
        </Button>
      </Space>
      <Space size="small" wrap>
        <Button onClick={resetYearFilter} type="text">
          {t("filters.reset_year")}
        </Button>
        {pendingRange && (
          <>
            <Button type="primary" onClick={applyPendingRange}>
              {t("filters.apply_range")}
            </Button>
            <Button onClick={cancelPendingRange}>{t("filters.cancel")}</Button>
          </>
        )}
      </Space>
      <div>
        <Text type="secondary">{t("filters.year_range")}</Text>
        <div style={{ paddingRight: 8 }}>
          <Slider
            range
            min={timelineMin}
            max={timelineMax}
            value={displayRange ?? undefined}
            onChange={(value) => handleFilterChange(subField, value as [number, number])}
          />
        </div>
      </div>
    </Space>
  );

  const legendPanel = (
    <Space direction="vertical" size="small" style={{ width: "100%" }}>
      <Tag className="soft-tag soft-tag-red">{t("legend.ccf")}</Tag>
      <Tag className="soft-tag soft-tag-indigo">{t("legend.global")}</Tag>
      <Tag className="soft-tag soft-tag-emerald">{t("legend.cnki")}</Tag>
      <Tag className="soft-tag soft-tag-slate">{t("legend.ref_only")}</Tag>
      <Tag className="soft-tag soft-tag-amber">
        {t("legend.size_by")} {sizeBy === "pagerank" ? t("size.pagerank") : t("size.citations")}
      </Tag>
      <Text type="secondary">{t("sync.queue")}</Text>
      <Space size="small" wrap>
        <Tag className="soft-tag soft-tag-indigo">
          {t("sync.queued")} {syncStatus.queued ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-amber">
          {t("sync.running")} {syncStatus.running ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-emerald">
          {t("sync.idle")} {syncStatus.idle ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-red">
          {t("sync.failed")} {syncStatus.failed ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-violet">
          jobs {Object.values(syncStatus.jobs || {}).reduce((acc, cur) => acc + Number(cur || 0), 0)}
        </Tag>
        <Tag className={`soft-tag ${(syncStatus.alerts_open ?? 0) > 0 ? "soft-tag-red" : "soft-tag-emerald"}`}>
          alerts {syncStatus.alerts_open ?? 0}
        </Tag>
      </Space>
    </Space>
  );

  const syncPanel = (
    <Space direction="vertical" size="small" style={{ width: "100%" }}>
      <Text type="secondary">{t("sync.queue")}</Text>
      <Space size="small" wrap>
        <Tag className="soft-tag soft-tag-indigo">
          {t("sync.queued")} {syncStatus.queued ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-amber">
          {t("sync.running")} {syncStatus.running ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-emerald">
          {t("sync.idle")} {syncStatus.idle ?? 0}
        </Tag>
        <Tag className="soft-tag soft-tag-red">
          {t("sync.failed")} {syncStatus.failed ?? 0}
        </Tag>
      </Space>
      <Space size="small" wrap>
        <Button
          onClick={async () => {
            setSyncBusy(true);
            try {
              const res = await enqueueAllSync();
              message.success(`${t("sync.enqueue_all")}: ${res.enqueued}`);
              refreshSyncStatus();
            } finally {
              setSyncBusy(false);
            }
          }}
          loading={syncBusy}
        >
          {t("sync.enqueue_all")}
        </Button>
        <Button
          onClick={async () => {
            setSyncBusy(true);
            try {
              const res = await runSync(syncLimit);
              message.success(
                `${t("sync.processed")} ${res.processed}, ${t("sync.failed")} ${res.failed}`
              );
              refreshSyncStatus();
            } finally {
              setSyncBusy(false);
            }
          }}
          loading={syncBusy}
        >
          {t("sync.run")}
        </Button>
        <InputNumber
          min={1}
          max={50}
          value={syncLimit}
          onChange={(value) => setSyncLimit(value ?? 5)}
        />
        <Button
          onClick={async () => {
            const res = await runQueuedJobs(syncLimit);
            message.success(`jobs ${res.executed}`);
            refreshJobBoard();
            refreshSyncStatus();
          }}
        >
          {t("workspace.jobs_run")}
        </Button>
      </Space>
      <Button
        onClick={async () => {
          const res = await cleanupCitations();
          message.success(`${t("sync.cleaned")} ${res.removed ?? 0}`);
          refreshSyncStatus();
        }}
      >
        {t("sync.cleanup")}
      </Button>
      <Text type="secondary" style={{ marginTop: 8 }}>
        {t("zotero.batch")}
      </Text>
      <Space size="small" wrap>
        <Button
          onClick={async () => {
            const res = await zoteroMatchAll(zoteroLimit);
            message.success(`${t("zotero.match_all")}: ${res.matched}`);
          }}
        >
          {t("zotero.match_all")}
        </Button>
        <Button
          onClick={async () => {
            const res = await zoteroPushAll(zoteroLimit);
            message.success(`${t("zotero.push_all")}: ${res.updated}`);
          }}
        >
          {t("zotero.push_all")}
        </Button>
        <Button
          onClick={async () => {
            const res = await zoteroSyncIds(zoteroLimit);
            message.success(`${t("zotero.sync_ids")}: ${res.synced}`);
          }}
        >
          {t("zotero.sync_ids")}
        </Button>
        <InputNumber
          min={1}
          max={100}
          value={zoteroLimit}
          onChange={(value) => setZoteroLimit(value ?? 20)}
        />
      </Space>
      <Space size="small" wrap>
        <Switch checked={autoPush} onChange={setAutoPush} />
        <Text type="secondary">{t("zotero.auto_push")}</Text>
      </Space>
      <Text type="secondary" style={{ marginTop: 8 }}>
        {t("backfill.title")}
      </Text>
      <Space size="small" wrap>
        <Button
          onClick={async () => {
            setSyncBusy(true);
            try {
              const res = await backfillPapers({
                limit: backfillLimit,
                summary: true,
                references: true,
                force: false
              });
              message.success(
                `${t("backfill.result")}: ${res.processed}, ${t("backfill.summaries")} ${res.summary_added}, ${t("backfill.refs")} ${res.references_parsed}, schema ${res.schemas_extracted ?? 0}, exp ${res.auto_experiments ?? 0}, cells ${res.metric_cells_upserted ?? 0}`
              );
              fetchData(subField, yearRange);
            } finally {
              setSyncBusy(false);
            }
          }}
          loading={syncBusy}
        >
          {t("backfill.run")}
        </Button>
        <InputNumber
          min={1}
          max={50}
          value={backfillLimit}
          onChange={(value) => setBackfillLimit(value ?? 5)}
        />
      </Space>
    </Space>
  );

  const detailsPanel = selected ? (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <div>
        <Title level={5} style={{ margin: 0 }}>
          {shortLabel(selected.label)}
        </Title>
        <Text type="secondary">{selected.authors || t("details.unknown_authors")}</Text>
      </div>
      <Space size="small" wrap>
        {selected.year && <Tag>{selected.year}</Tag>}
        {selected.sub_field && <Tag className="soft-tag soft-tag-indigo">{selected.sub_field}</Tag>}
        {!selected.sub_field && selected.open_sub_field && <Tag className="soft-tag soft-tag-violet">{selected.open_sub_field}</Tag>}
        {selected.ccf_level && <Tag className="soft-tag soft-tag-red">CCF {selected.ccf_level}</Tag>}
      </Space>
      <Space size="small" wrap>
        <Button
          size="small"
          onClick={async () => {
            if (!selected) return;
            const next = selected.read_status === 1 ? 0 : 1;
            const updated = await updatePaper(selected.id, { read_status: next });
            setSelected({ ...selected, read_status: updated.read_status ?? next });
            message.success(next === 1 ? t("msg.mark_read") : t("msg.mark_unread"));
            fetchData(subField, yearRange);
            await maybeAutoPush(updated.id, selected.zotero_item_key);
          }}
        >
          {selected.read_status === 1 ? t("details.mark_unread") : t("details.mark_read")}
        </Button>
        <Button size="small" onClick={() => setInspectorTab("chat")}>
          {t("qa.title")}
        </Button>
        {!readingMode && (
          <>
            <Button size="small" onClick={() => addToComparison(selected.id)}>
              {t("compare.add")}
            </Button>
            <Button size="small" onClick={() => setPathSourceId(selected.id)}>
              {t("compare.set_source")}
            </Button>
            <Button size="small" onClick={() => setPathTargetId(selected.id)}>
              {t("compare.set_target")}
            </Button>
            <Button size="small" onClick={loadNeighbors}>
              {t("details.load_neighbors")}
            </Button>
          </>
        )}
      </Space>
      {!readingMode && (
        <div>
          <Text type="secondary">{t("details.sub_field")}</Text>
          <Select
            allowClear
            style={{ width: "100%", marginTop: 6 }}
            options={subFieldOptions}
            value={selected.sub_field ?? undefined}
            onChange={async (value) => {
              if (!selected) return;
              const updated = await updatePaper(selected.id, { sub_field: value ?? null });
              setSelected({ ...selected, sub_field: updated.sub_field ?? null });
              message.success(t("msg.subfield_updated"));
              await maybeAutoPush(updated.id, selected.zotero_item_key);
            }}
            placeholder={t("details.sub_field")}
          />
        </div>
      )}
      <Paragraph style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
        {selected.abstract || t("details.no_abstract")}
      </Paragraph>
      {selected.summary_one && (
        <div>
          <Text type="secondary">{t("details.summary")}</Text>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            {selected.summary_one}
          </Paragraph>
        </div>
      )}
      <Space size="small" wrap>
        {selected.citation_count !== null && (
          <Tag className="soft-tag soft-tag-amber">
            {t("details.citations")}: {selected.citation_count ?? 0}
          </Tag>
        )}
        {selected.reference_count !== null && (
          <Tag className="soft-tag soft-tag-violet">
            {t("details.references")}: {selected.reference_count ?? 0}
          </Tag>
        )}
      </Space>
      {!readingMode && (
        <>
          <div>
            <Text type="secondary">{t("details.zotero_key")}</Text>
            <Input
              value={zoteroKey}
              onChange={(e) => setZoteroKey(e.target.value)}
              placeholder="e.g. ABCD1234"
              style={{ marginTop: 6 }}
            />
            {selected.zotero_item_id && (
              <Text type="secondary" style={{ display: "block", marginTop: 6 }}>
                {t("details.zotero_id")}: {selected.zotero_item_id} ({selected.zotero_library ?? "user"})
              </Text>
            )}
          </div>
          <Space size="small" wrap>
            <Button
              onClick={async () => {
                if (!selected) return;
                const updated = await updatePaper(selected.id, { zotero_item_key: zoteroKey });
                setSelected({
                  ...selected,
                  zotero_item_key: updated.zotero_item_key ?? null,
                  zotero_library: updated.zotero_library ?? null,
                  zotero_item_id: updated.zotero_item_id ?? null
                });
                message.success(t("msg.zotero_saved"));
                await maybeAutoPush(updated.id, updated.zotero_item_key ?? null);
              }}
            >
              {t("details.save")}
            </Button>
            <Button
              onClick={async () => {
                if (!selected) return;
                try {
                  const updated = await matchZotero(selected.id);
                  setSelected({
                    ...selected,
                    zotero_item_key: updated.zotero_item_key ?? null,
                    zotero_library: updated.zotero_library ?? null,
                    zotero_item_id: updated.zotero_item_id ?? null
                  });
                  setZoteroKey(updated.zotero_item_key ?? "");
                  message.success(t("msg.zotero_matched"));
                } catch (err: any) {
                  message.error(err?.response?.data?.detail || t("msg.zotero_match_failed"));
                }
              }}
            >
              {t("details.match")}
            </Button>
            <Button
              onClick={async () => {
                if (!selected) return;
                try {
                  await pushZotero(selected.id);
                  message.success(t("msg.zotero_pushed"));
                } catch (err: any) {
                  message.error(err?.response?.data?.detail || t("msg.zotero_update_failed"));
                }
              }}
            >
              {t("details.push")}
            </Button>
            <Button
              type="primary"
              onClick={() => {
                if (!selected?.zotero_item_key) {
                  message.warning(t("msg.zotero_key_required"));
                  return;
                }
                const library = selected.zotero_library || "";
                let url = `zotero://select/library/items/${selected.zotero_item_key}`;
                if (library.startsWith("group:")) {
                  const groupId = library.split(":")[1];
                  if (groupId) {
                    url = `zotero://select/groups/${groupId}/items/${selected.zotero_item_key}`;
                  }
                }
                window.location.href = url;
              }}
            >
              {t("details.open")}
            </Button>
          </Space>
        </>
      )}
    </Space>
  ) : (
    <Text type="secondary">{t("details.select_hint")}</Text>
  );

  const chatPanel = (
    <Space direction="vertical" size="small" style={{ width: "100%" }}>
      <Space size="small" wrap>
        <Select
          allowClear
          style={{ width: 200 }}
          value={chatSessionId}
          placeholder={t("qa.session")}
          options={chatSessions.map((session) => ({
            label: `#${session.id} ${session.title || ""}`.trim(),
            value: session.id
          }))}
          onChange={(value) => setChatSessionId(value)}
        />
        <Button
          onClick={async () => {
            const session = await createChatSession({ title: qaQuery.slice(0, 30), language: "zh" });
            setChatSessionId(session.id);
            await refreshChatSessions();
          }}
        >
          {t("qa.new_session")}
        </Button>
      </Space>
      <Select
        value={qaMode}
        onChange={(value) => setQaMode(value as "single" | "compare")}
        options={[
          { label: t("qa.single"), value: "single" },
          { label: t("qa.multi"), value: "compare" }
        ]}
      />
      <Input.TextArea
        value={qaQuery}
        onChange={(e) => setQaQuery(e.target.value)}
        placeholder={t("qa.placeholder")}
        autoSize={{ minRows: 2, maxRows: 4 }}
      />
      <Button type="primary" onClick={askPaperQa} loading={qaLoading}>
        {t("qa.ask")}
      </Button>
      <List
        size="small"
        dataSource={chatHistory.slice(-8)}
        locale={{ emptyText: t("qa.history_empty") }}
        renderItem={(msg) => (
          <List.Item>
            <Space direction="vertical" size={0}>
              <Text strong>{msg.role === "assistant" ? "AI" : "You"}</Text>
              <Text type="secondary">{msg.content.slice(0, 220)}</Text>
            </Space>
          </List.Item>
        )}
      />
      {qaResult && (
        <>
          <Paragraph style={{ whiteSpace: "pre-wrap" }}>{qaResult.answer}</Paragraph>
          {typeof qaResult.trace_score === "number" && (
            <Tag color={qaResult.trace_score >= 0.6 ? "green" : qaResult.trace_score >= 0.4 ? "gold" : "red"}>
              Traceability: {qaResult.trace_score.toFixed(2)}
            </Tag>
          )}
          <List
            size="small"
            header={<Text type="secondary">{t("qa.sources")}</Text>}
            dataSource={qaResult.sources}
            renderItem={(source) => (
              <List.Item>
                <Text type="secondary">
                  #{source.paper_id} {source.title}{" "}
                  {source.chunk_index !== undefined ? ` [chunk ${source.chunk_index}]` : ""}
                </Text>
              </List.Item>
            )}
          />
        </>
      )}
    </Space>
  );

  const advancedControls = (
    <Space direction="vertical" size="middle" className="graph-advanced-controls">
      <div className="graph-advanced-row">
        <Text type="secondary">{t("graph.focus")}</Text>
        <Switch checked={focusMode} onChange={setFocusMode} />
      </div>
      <div className="graph-advanced-row">
        <Text type="secondary">{t("graph.smart_declutter")}</Text>
        <Switch checked={smartDeclutter} onChange={setSmartDeclutter} />
      </div>
      <div className="graph-advanced-row">
        <Text type="secondary">{t("graph.cluster_hull")}</Text>
        <Switch checked={showClusterHull} onChange={setShowClusterHull} />
      </div>
      <div className="graph-advanced-row">
        <Text type="secondary">{t("graph.edge_bundling")}</Text>
        <Switch checked={edgeBundling} onChange={setEdgeBundling} />
      </div>
      <div className="graph-advanced-row">
        <Text type="secondary">{t("graph.labels")}</Text>
        <Switch checked={showLabels} onChange={setShowLabels} />
      </div>
      <div>
        <Text type="secondary">{t("graph.intent_all")}</Text>
        <Select
          value={edgeIntentFilter}
          style={{ width: "100%", marginTop: 8 }}
          onChange={(value) =>
            setEdgeIntentFilter(value as "all" | "build_on" | "contrast" | "use_as_baseline" | "mention")
          }
          options={[
            { label: t("graph.intent_all"), value: "all" },
            { label: t("graph.intent_build_on"), value: "build_on" },
            { label: t("graph.intent_contrast"), value: "contrast" },
            { label: t("graph.intent_baseline"), value: "use_as_baseline" },
            { label: t("graph.intent_mention"), value: "mention" }
          ]}
        />
      </div>
      <div>
        <Text type="secondary">{t("graph.label_auto")}</Text>
        <Select
          value={labelMode}
          style={{ width: "100%", marginTop: 8 }}
          onChange={(value) => setLabelMode(value as "auto" | "selected" | "all")}
          options={[
            { label: t("graph.label_auto"), value: "auto" },
            { label: t("graph.label_selected"), value: "selected" },
            { label: t("graph.label_all"), value: "all" }
          ]}
        />
      </div>
      <Button block onClick={classifyIntents}>
        {t("graph.classify_intent")}
      </Button>
      {!selected && <Text type="secondary">{t("graph.relation_hint")}</Text>}
    </Space>
  );

  return (
    <div className={`graph-shell ${standalone ? "is-standalone" : ""} ${sidebarCollapsed ? "is-sidebar-collapsed" : ""}`}>
      <div className="graph-floating-brand">
        <Text strong>PaperTrail</Text>
        <Button size="small" onClick={() => (window.location.href = "/")}>
          {t("nav.home")}
        </Button>
      </div>
      <div className={`graph-sidebar ${sidebarCollapsed ? "is-collapsed" : ""}`}>
        <Card className="graph-card graph-sidebar-card" size="small">
          <div className="graph-sidebar-header">
            <Tooltip title={sidebarCollapsed ? t("graph.open_panel") : t("graph.maximize_graph")} placement="right">
              <Button
                type="text"
                icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setSidebarCollapsed((prev) => !prev)}
              />
            </Tooltip>
          </div>
          {sidebarCollapsed ? (
            <Space direction="vertical" size="small" className="graph-sidebar-icons">
              <Tooltip title={t("filters.title")} placement="right">
                <Button
                  type={sidebarTab === "filters" ? "primary" : "text"}
                  icon={<FilterOutlined />}
                  onClick={() => {
                    setSidebarTab("filters");
                    setSidebarCollapsed(false);
                  }}
                />
              </Tooltip>
              <Tooltip title={t("legend.title")} placement="right">
                <Button
                  type={sidebarTab === "legend" ? "primary" : "text"}
                  icon={<AppstoreOutlined />}
                  onClick={() => {
                    setSidebarTab("legend");
                    setSidebarCollapsed(false);
                  }}
                />
              </Tooltip>
              <Tooltip title={t("sync.title")} placement="right">
                <Button
                  type={sidebarTab === "sync" ? "primary" : "text"}
                  icon={<SyncOutlined />}
                  onClick={() => {
                    setSidebarTab("sync");
                    setSidebarCollapsed(false);
                  }}
                />
              </Tooltip>
            </Space>
          ) : (
            <Tabs
              size="small"
              activeKey={sidebarTab}
              onChange={(key) => setSidebarTab(key as "filters" | "legend" | "sync")}
              items={[
                { key: "filters", label: t("filters.title"), children: filterPanel },
                { key: "legend", label: t("legend.title"), children: legendPanel },
                { key: "sync", label: t("sync.title"), children: syncPanel }
              ]}
            />
          )}
        </Card>
      </div>
      <div className="graph-main">
        <Card className="graph-card graph-topbar" size="small">
          <div className="graph-topbar-row">
            <Space size="small" wrap>
              <Button
                onClick={() => {
                  if (window.innerWidth <= 1024) {
                    setSidebarDrawerOpen(true);
                    return;
                  }
                  setSidebarCollapsed((prev) => !prev);
                }}
              >
                {sidebarCollapsed ? t("graph.open_panel") : t("graph.maximize_graph")}
              </Button>
              <Text type="secondary">{t("graph.relation_mode")}</Text>
              <Segmented
                className="graph-relation-segment"
                value={edgeFocus}
                options={[
                  { label: t("details.show_refs"), value: "out" },
                  { label: t("details.show_cited"), value: "in" },
                  { label: t("details.show_all"), value: "all" }
                ]}
                onChange={(value) => {
                  setEdgeFocus(value as "all" | "out" | "in");
                  if (selected) loadNeighbors();
                }}
              />
            </Space>
            <Space size="small" wrap>
              <Tag className="soft-tag soft-tag-indigo">
                {t("sync.queued")} {syncStatus.queued ?? 0}
              </Tag>
              <Tag className="soft-tag soft-tag-amber">
                {t("sync.running")} {syncStatus.running ?? 0}
              </Tag>
              <Tag className="soft-tag soft-tag-red">
                {t("sync.failed")} {syncStatus.failed ?? 0}
              </Tag>
              <Tag className="soft-tag soft-tag-violet">
                {t("graph.community_count")} {communityCount}
              </Tag>
              <Popover trigger="click" placement="bottomRight" content={advancedControls}>
                <Button icon={<SettingOutlined />}>{t("filters.title")}</Button>
              </Popover>
            </Space>
          </div>
        </Card>
        <div className="graph-canvas">
          {loading && (
            <div className="graph-loading">
              <Spin />
            </div>
          )}
          {layout === "timeline" && timelineGuides.length > 0 && (
            <div className="graph-year-guides">
              {timelineGuides.map((guide) => (
                <div key={guide.year} className="graph-year-guide" style={{ left: guide.x }}>
                  <span>{guide.year}</span>
                </div>
              ))}
            </div>
          )}
          {edgeBundling && bundledPaths.length > 0 && (
            <svg
              className="graph-edge-bundles"
              viewBox={`0 0 ${canvasSize.width} ${canvasSize.height}`}
              preserveAspectRatio="none"
            >
              {bundledPaths.map((bundle) => (
                <g key={bundle.id}>
                  <path
                    d={bundle.path}
                    fill="none"
                    stroke={bundle.stroke}
                    strokeWidth={bundle.width}
                    opacity={bundle.opacity}
                    className="graph-edge-bundle-path"
                  />
                  {bundle.count >= 8 && (
                    <text x={bundle.labelX} y={bundle.labelY} className="graph-edge-bundle-label">
                      {bundle.count}
                    </text>
                  )}
                </g>
              ))}
            </svg>
          )}
          {showClusterHull && clusterHulls.length > 0 && (
            <svg
              className="graph-cluster-hulls"
              viewBox={`0 0 ${canvasSize.width} ${canvasSize.height}`}
              preserveAspectRatio="none"
            >
              {clusterHulls.map((cluster) => (
                <g key={cluster.id}>
                  <path
                    d={cluster.path}
                    fill={cluster.fill}
                    stroke={cluster.stroke}
                    strokeWidth={1.2}
                    className="graph-cluster-hull-path"
                  />
                  <text x={cluster.labelX} y={cluster.labelY} className="graph-cluster-hull-label">
                    {cluster.label} · {cluster.size}
                  </text>
                </g>
              ))}
            </svg>
          )}
          <div ref={containerRef} className="graph-cytoscape" />
          <div className={`graph-floating-capsule ${!pathSourceId && !pathTargetId ? "is-idle" : ""}`}>
            <Space size="small" wrap>
              <Text type="secondary">{t("graph.path_context_hint")}</Text>
              <Select
                allowClear
                showSearch
                style={{ width: 210 }}
                placeholder={t("graph.path_source")}
                value={pathSourceId}
                options={pathNodeOptions}
                onChange={(value) => setPathSourceId(value)}
              />
              <Select
                allowClear
                showSearch
                style={{ width: 210 }}
                placeholder={t("graph.path_target")}
                value={pathTargetId}
                options={pathNodeOptions}
                onChange={(value) => setPathTargetId(value)}
              />
              <Button icon={<AimOutlined />} type="primary" loading={pathLoading} onClick={highlightShortestPath}>
                {t("graph.shortest_path")}
              </Button>
              <Button
                icon={<DeleteOutlined />}
                onClick={() => {
                  if (!cyRef.current) return;
                  cyRef.current.nodes().removeClass("path-step");
                  cyRef.current.edges().removeClass("path-edge");
                  setPathSourceId(undefined);
                  setPathTargetId(undefined);
                }}
              >
                {t("graph.clear_shortest_path")}
              </Button>
              <Tag className="soft-tag soft-tag-indigo">
                {t("graph.path_source")}: {pathSourceId || "-"}
              </Tag>
              <Tag className="soft-tag soft-tag-violet">
                {t("graph.path_target")}: {pathTargetId || "-"}
              </Tag>
            </Space>
          </div>
          <div className="graph-floating-timeline">
            {bubbleLabel && (
              <div className="timeline-bubble">
                <span className="timeline-bubble-label">
                  {isDragging ? t("timeline.preview") : t("timeline.pending")}
                </span>
                <span className="timeline-bubble-range">{bubbleLabel}</span>
              </div>
            )}
            <Text type="secondary" className="timeline-title">
              {t("timeline.title")}
            </Text>
            <div
              className={`timeline ${isDragging ? "is-dragging" : ""}`}
              ref={timelineRef}
              onMouseDown={handleTimelineMouseDown}
              onMouseMove={handleTimelineMouseMove}
              onMouseUp={handleTimelineMouseUp}
              onMouseLeave={handleTimelineMouseUp}
            >
              {timelineData.map((item) => {
                const isInRange = displayRange && item.year >= displayRange[0] && item.year <= displayRange[1];
                return (
                  <div
                    key={item.year}
                    className={`timeline-item ${isInRange ? "is-selected" : ""} ${isDragging ? "is-dragging" : ""}`}
                    data-year={item.year}
                  >
                    <div className="timeline-bar" style={{ height: `${10 + item.count * 6}px` }} />
                    <div className="timeline-year">{item.year}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
      <div className={`graph-inspector ${inspectorCollapsed ? "is-collapsed" : ""}`}>
        {inspectorCollapsed ? (
          <Button
            className="graph-inspector-collapse-btn"
            icon={<LeftOutlined />}
            onClick={() => setInspectorCollapsed(false)}
          />
        ) : (
          <Card className="graph-card graph-inspector-card" size="small">
            <Tabs
              size="small"
              activeKey={inspectorTab}
              onChange={(key) => setInspectorTab(key as "details" | "chat")}
              tabBarExtraContent={
                <Space size={6} className="inspector-reading-toggle">
                  <Text type="secondary">{t("inspector.reading_mode")}</Text>
                  <Switch size="small" checked={readingMode} onChange={setReadingMode} />
                  <Button
                    type="text"
                    icon={<RightOutlined />}
                    onClick={() => setInspectorCollapsed(true)}
                  />
                </Space>
              }
              items={[
                {
                  key: "details",
                  label: t("details.title"),
                  children: detailsPanel
                },
                {
                  key: "chat",
                  label: t("qa.title"),
                  children: chatPanel
                }
              ]}
            />
          </Card>
        )}
      </div>
      {menu.visible && menu.node && (
        <div
          className="graph-context-menu"
          style={{ left: menu.x, top: menu.y }}
          onMouseLeave={() => setMenu((prev) => ({ ...prev, visible: false }))}
        >
          <Button
            type="text"
            block
            onClick={() => {
              addToComparison(menu.node!.id);
              setMenu((prev) => ({ ...prev, visible: false }));
            }}
          >
            {t("compare.add")}
          </Button>
          <Button
            type="text"
            block
            onClick={() => {
              setPathSourceId(menu.node!.id);
              setMenu((prev) => ({ ...prev, visible: false }));
            }}
          >
            {t("compare.set_source")}
          </Button>
          <Button
            type="text"
            block
            onClick={() => {
              setPathTargetId(menu.node!.id);
              setMenu((prev) => ({ ...prev, visible: false }));
            }}
          >
            {t("compare.set_target")}
          </Button>
          <Button
            type="text"
            block
            onClick={async () => {
              if (!pathSourceId) {
                setPathSourceId(menu.node!.id);
              } else {
                setPathTargetId(menu.node!.id);
                window.setTimeout(() => {
                  highlightShortestPath();
                }, 0);
              }
              setMenu((prev) => ({ ...prev, visible: false }));
            }}
          >
            {t("graph.shortest_path")}
          </Button>
        </div>
      )}
      <Drawer
        title={t("graph.panel")}
        placement="left"
        width={360}
        open={sidebarDrawerOpen}
        onClose={() => setSidebarDrawerOpen(false)}
        extra={
          <Button
            size="small"
            onClick={() => {
              setSidebarCollapsed(false);
              setSidebarDrawerOpen(false);
            }}
          >
            {t("graph.pin_panel")}
          </Button>
        }
      >
        <Tabs
          size="small"
          activeKey={sidebarTab}
          onChange={(key) => setSidebarTab(key as "filters" | "legend" | "sync")}
          items={[
            { key: "filters", label: t("filters.title"), children: filterPanel },
            { key: "legend", label: t("legend.title"), children: legendPanel },
            { key: "sync", label: t("sync.title"), children: syncPanel }
          ]}
        />
      </Drawer>
      <Modal
        title={t("manage.title")}
        open={subfieldModalOpen}
        onCancel={() => setSubfieldModalOpen(false)}
        footer={null}
      >
        <Space direction="vertical" style={{ width: "100%" }}>
          <Space.Compact style={{ width: "100%" }}>
            <Input
              placeholder={t("manage.new_name")}
              value={newSubfieldName}
              onChange={(e) => setNewSubfieldName(e.target.value)}
            />
            <Button type="primary" onClick={handleCreateSubfield}>
              {t("manage.add")}
            </Button>
          </Space.Compact>
          <Space size="small" wrap>
            <Button
              loading={openTagLoading}
              onClick={async () => {
                setOpenTagLoading(true);
                try {
                  const result = await discoverOpenTags({ limit: 300, add_to_subfields: false });
                  setOpenTagCandidates(result.candidates);
                  message.success(`${t("manage.discover_ok")}: ${result.candidates.length}`);
                } finally {
                  setOpenTagLoading(false);
                }
              }}
            >
              {t("manage.discover_tags")}
            </Button>
            <Button
              loading={openTagLoading}
              onClick={async () => {
                setOpenTagLoading(true);
                try {
                  const result = await discoverOpenTags({ limit: 300, add_to_subfields: true });
                  message.success(`${t("manage.added_ok")}: ${result.added}`);
                  setOpenTagCandidates(result.candidates);
                  await loadSubfields(true);
                  await loadSubfields(false);
                } finally {
                  setOpenTagLoading(false);
                }
              }}
            >
              {t("manage.add_discovered")}
            </Button>
          </Space>
          {openTagCandidates.length > 0 && (
            <Space size="small" wrap>
              {openTagCandidates.slice(0, 20).map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </Space>
          )}
          <List
            dataSource={allSubfields}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Switch
                    key="active"
                    checked={item.active === 1}
                    onChange={() => handleToggleSubfield(item)}
                  />,
                  <Button key="delete" type="link" danger onClick={() => handleDeleteSubfield(item)}>
                    {t("manage.delete")}
                  </Button>
                ]}
              >
                {item.name}
              </List.Item>
            )}
          />
        </Space>
      </Modal>
    </div>
  );
}
