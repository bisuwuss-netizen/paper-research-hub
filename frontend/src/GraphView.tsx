import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import cytoscape from "cytoscape";
import {
  Card,
  Select,
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
  Drawer
} from "antd";
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
  discoverOpenTags,
  type GraphNode,
  type GraphResponse,
  type Subfield,
  type SyncStatus,
  type PaperNote,
  type ReadingTask,
  type Experiment,
  type ComparePapersResponse,
  type ChatWithPapersResponse
} from "./api";

const { Title, Text, Paragraph } = Typography;

type GraphViewProps = {
  t: (key: string) => string;
  standalone?: boolean;
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
  const [syncStatus, setSyncStatus] = useState<SyncStatus>({});
  const [syncLimit, setSyncLimit] = useState(5);
  const [syncBusy, setSyncBusy] = useState(false);
  const [zoteroLimit, setZoteroLimit] = useState(20);
  const [autoPush, setAutoPush] = useState(false);
  const [backfillLimit, setBackfillLimit] = useState(5);
  const [edgeMinConfidence, setEdgeMinConfidence] = useState<number>(0);
  const [sidebarTab, setSidebarTab] = useState<"filters" | "legend" | "sync">("filters");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarDrawerOpen, setSidebarDrawerOpen] = useState(false);
  const [note, setNote] = useState<PaperNote | null>(null);
  const [noteSaving, setNoteSaving] = useState(false);
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

  const applyLayout = (layoutType: "force" | "hierarchy" | "timeline") => {
    const cy = cyRef.current;
    if (!cy) return;
    if (layoutType === "force") {
      cy.layout({ name: "cose", animate: true, padding: 30 }).run();
      return;
    }
    if (layoutType === "hierarchy") {
      cy.layout({ name: "breadthfirst", directed: true, padding: 30, spacingFactor: 1.2 }).run();
      return;
    }
    const nodes = cy.nodes();
    const years = nodes
      .map((n: cytoscape.NodeSingular) => n.data("full")?.year)
      .filter(Boolean) as number[];
    const minYear = Math.min(...years, 2000);
    const maxYear = Math.max(...years, 2026);
    const width = containerRef.current?.clientWidth ?? 800;
    const height = containerRef.current?.clientHeight ?? 600;
    const positions: Record<string, { x: number; y: number }> = {};
    const groups: Record<number, string[]> = {};
    nodes.forEach((n: cytoscape.NodeSingular) => {
      const year = n.data("full")?.year ?? minYear;
      if (!groups[year]) groups[year] = [];
      groups[year].push(n.id());
    });
    Object.keys(groups).forEach((yearStr) => {
      const year = Number(yearStr);
      const list = groups[year];
      const x =
        60 +
        ((year - minYear) / Math.max(1, maxYear - minYear)) * (width - 120);
      list.forEach((id, idx) => {
        const y = 50 + (idx % 20) * 24 + Math.floor(idx / 20) * 10;
        positions[id] = { x, y: Math.min(y, height - 60) };
      });
    });
    cy.layout({ name: "preset", positions }).run();
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

  useEffect(() => {
    refreshSyncStatus();
    const id = window.setInterval(refreshSyncStatus, 10000);
    return () => window.clearInterval(id);
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
    const hide = () => setMenu((prev) => ({ ...prev, visible: false }));
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, []);

  const loadNodeWorkspace = async (paperId: number) => {
    try {
      const [noteData, taskData, experimentData] = await Promise.all([
        fetchPaperNotes(paperId),
        fetchTasks({ paper_id: paperId }),
        fetchExperiments({ paper_id: paperId })
      ]);
      setNote(noteData);
      setTasks(taskData);
      setExperiments(experimentData);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (!selected) {
      setNote(null);
      setTasks([]);
      setExperiments([]);
      return;
    }
    loadNodeWorkspace(selected.id);
    if (focusMode) {
      loadNeighbors();
    }
  }, [selected?.id, focusMode]);

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
              "font-size": 8,
              "text-wrap": "wrap",
              "text-max-width": "110",
              "min-zoomed-font-size": 6,
              width: "data(size)",
              height: "data(size)",
              "text-valign": "top",
              "text-halign": "center",
              "text-margin-y": -8,
              "text-background-color": "#ffffff",
              "text-background-opacity": 0.78,
              "text-background-padding": "2",
              "text-opacity": 0.95
            }
          },
          {
            selector: "edge",
            style: {
              width: (edge: cytoscape.EdgeSingular) =>
                0.6 + (edge.data("confidence") ?? 0.7) * 2.2,
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "#94a3b8",
              "line-color": "#cbd5f5",
              opacity: (edge: cytoscape.EdgeSingular) =>
                Math.max(0.2, Math.min(0.95, (edge.data("confidence") ?? 0.7) * 0.95))
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
  }, [edgeFocus, edgeIntentFilter, focusMode, selected, graph]);

  useEffect(() => {
    fetchData(subField, yearRange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sizeBy, edgeRecentYears, minIndegree, lazyMode, nodeLimit, sortBy, uploadedOnly, edgeMinConfidence]);

  useEffect(() => {
    if (!graph || !cyRef.current) return;
    const cy = cyRef.current;
    cy.elements().remove();

    const elements: cytoscape.ElementDefinition[] = [];
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
    updateNodeLabels();
    applyLayout(layout);
    clearPathAnimation();
  }, [graph, showLabels]);

  useEffect(() => {
    applyLayout(layout);
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
          (total <= 50 && zoom >= 0.7) ||
          (total <= 120 && zoom >= 1.0) ||
          zoom >= 1.3;
      }
      node.data("displayLabel", shouldShow ? base : "");
      node.data("labelOpacity", shouldShow ? 0.95 : 0);
      node.data("labelBgOpacity", shouldShow ? 0.78 : 0);
    });
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
      applyLayout(layout);
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
      await streamChatWithPapers(
        {
          query,
          paper_ids: paperIds,
          top_k: 8,
          language: navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en"
        },
        {
          onMeta: (meta) =>
            setQaResult((prev) => ({
              query: meta.query,
              paper_ids: meta.paper_ids,
              answer: prev?.answer ?? "",
              sources: prev?.sources ?? [],
              context_count: meta.context_count
            })),
          onDelta: (delta) =>
            setQaResult((prev) => ({
              query,
              paper_ids: paperIds,
              answer: `${prev?.answer || ""}${delta}`,
              sources: prev?.sources || [],
              context_count: prev?.context_count || 0
            })),
          onSources: (sources) =>
            setQaResult((prev) => ({
              query,
              paper_ids: paperIds,
              answer: prev?.answer || "",
              sources,
              context_count: prev?.context_count || 0
            }))
        }
      );
    } catch {
      // Fallback to non-streaming API when SSE is unavailable.
      const result = await chatWithPapers({
        query,
        paper_ids: paperIds,
        top_k: 8,
        language: navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en"
      });
      setQaResult(result);
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
      <Tag color="red">{t("legend.ccf")}</Tag>
      <Tag color="blue">{t("legend.global")}</Tag>
      <Tag color="green">{t("legend.cnki")}</Tag>
      <Tag color="default">{t("legend.ref_only")}</Tag>
      <Tag color="gold">
        {t("legend.size_by")} {sizeBy === "pagerank" ? t("size.pagerank") : t("size.citations")}
      </Tag>
      <Text type="secondary">{t("sync.queue")}</Text>
      <Space size="small" wrap>
        <Tag color="blue">
          {t("sync.queued")} {syncStatus.queued ?? 0}
        </Tag>
        <Tag color="gold">
          {t("sync.running")} {syncStatus.running ?? 0}
        </Tag>
        <Tag color="green">
          {t("sync.idle")} {syncStatus.idle ?? 0}
        </Tag>
        <Tag color="red">
          {t("sync.failed")} {syncStatus.failed ?? 0}
        </Tag>
      </Space>
    </Space>
  );

  const syncPanel = (
    <Space direction="vertical" size="small" style={{ width: "100%" }}>
      <Text type="secondary">{t("sync.queue")}</Text>
      <Space size="small" wrap>
        <Tag color="blue">
          {t("sync.queued")} {syncStatus.queued ?? 0}
        </Tag>
        <Tag color="gold">
          {t("sync.running")} {syncStatus.running ?? 0}
        </Tag>
        <Tag color="green">
          {t("sync.idle")} {syncStatus.idle ?? 0}
        </Tag>
        <Tag color="red">
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
                `${t("backfill.result")}: ${res.processed}, ${t("backfill.summaries")} ${res.summary_added}, ${t("backfill.refs")} ${res.references_parsed}`
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

  return (
    <div className={`graph-shell ${standalone ? "is-standalone" : ""} ${sidebarCollapsed ? "is-sidebar-collapsed" : ""}`}>
      {!sidebarCollapsed && (
        <div className="graph-sidebar">
          <Card className="graph-card graph-sidebar-card" size="small">
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
          </Card>
        </div>
      )}
      <div className="graph-main">
        <Card className="graph-card graph-topbar" size="small">
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            <Space size="small" wrap>
              <Button
                onClick={() => {
                  if (sidebarCollapsed) {
                    setSidebarDrawerOpen(true);
                  } else {
                    setSidebarCollapsed(true);
                  }
                }}
              >
                {sidebarCollapsed ? t("graph.open_panel") : t("graph.maximize_graph")}
              </Button>
              <Text type="secondary">{t("graph.relation_mode")}</Text>
              <Button
                type={edgeFocus === "out" ? "primary" : "default"}
                onClick={() => {
                  const next = edgeFocus === "out" ? "all" : "out";
                  setEdgeFocus(next);
                  if (selected) loadNeighbors();
                }}
              >
                {t("details.show_refs")}
              </Button>
              <Button
                type={edgeFocus === "in" ? "primary" : "default"}
                onClick={() => {
                  const next = edgeFocus === "in" ? "all" : "in";
                  setEdgeFocus(next);
                  if (selected) loadNeighbors();
                }}
              >
                {t("details.show_cited")}
              </Button>
              {edgeFocus !== "all" && (
                <Button onClick={() => setEdgeFocus("all")}>{t("details.show_all")}</Button>
              )}
              <Switch checked={focusMode} onChange={setFocusMode} />
              <Text type="secondary">{t("graph.focus")}</Text>
              <Select
                value={edgeIntentFilter}
                style={{ width: 170 }}
                onChange={(value) =>
                  setEdgeIntentFilter(
                    value as "all" | "build_on" | "contrast" | "use_as_baseline" | "mention"
                  )
                }
                options={[
                  { label: t("graph.intent_all"), value: "all" },
                  { label: t("graph.intent_build_on"), value: "build_on" },
                  { label: t("graph.intent_contrast"), value: "contrast" },
                  { label: t("graph.intent_baseline"), value: "use_as_baseline" },
                  { label: t("graph.intent_mention"), value: "mention" }
                ]}
              />
              <Button onClick={classifyIntents}>{t("graph.classify_intent")}</Button>
              <Switch checked={showLabels} onChange={setShowLabels} />
              <Text type="secondary">{t("graph.labels")}</Text>
              <Select
                value={labelMode}
                style={{ width: 120 }}
                onChange={(value) => setLabelMode(value as "auto" | "selected" | "all")}
                options={[
                  { label: t("graph.label_auto"), value: "auto" },
                  { label: t("graph.label_selected"), value: "selected" },
                  { label: t("graph.label_all"), value: "all" }
                ]}
              />
              {!selected && <Text type="secondary">{t("graph.relation_hint")}</Text>}
            </Space>
            <Space size="small" wrap>
              <Select
                allowClear
                placeholder={t("graph.path_source")}
                style={{ width: 160 }}
                value={pathSourceId}
                onChange={(value) => setPathSourceId(value)}
                options={(graph?.nodes || []).map((n) => ({ label: shortLabel(n.label), value: n.id }))}
              />
              <Select
                allowClear
                placeholder={t("graph.path_target")}
                style={{ width: 160 }}
                value={pathTargetId}
                onChange={(value) => setPathTargetId(value)}
                options={(graph?.nodes || []).map((n) => ({ label: shortLabel(n.label), value: n.id }))}
              />
              <Button loading={pathLoading} onClick={highlightShortestPath}>
                {t("graph.shortest_path")}
              </Button>
              <Button
                onClick={() => {
                  if (!cyRef.current) return;
                  cyRef.current.nodes().removeClass("path-step");
                  cyRef.current.edges().removeClass("path-edge");
                }}
              >
                {t("graph.clear_shortest_path")}
              </Button>
            </Space>
            <Space size="small" wrap>
              <Tag color="red">{t("legend.ccf")}</Tag>
              <Tag color="blue">{t("legend.global")}</Tag>
              <Tag color="green">{t("legend.cnki")}</Tag>
              <Tag color="default">{t("legend.ref_only")}</Tag>
              <Tag color="green">{t("graph.intent_build_on")}</Tag>
              <Tag color="red">{t("graph.intent_contrast")}</Tag>
              <Tag color="gold">{t("graph.intent_baseline")}</Tag>
              <Tag color="blue">
                {t("sync.queued")} {syncStatus.queued ?? 0}
              </Tag>
              <Tag color="gold">
                {t("sync.running")} {syncStatus.running ?? 0}
              </Tag>
              <Tag color="green">
                {t("sync.idle")} {syncStatus.idle ?? 0}
              </Tag>
              <Tag color="red">
                {t("sync.failed")} {syncStatus.failed ?? 0}
              </Tag>
            </Space>
          </Space>
        </Card>
        <div className="graph-canvas">
          {loading && (
            <div className="graph-loading">
              <Spin />
            </div>
          )}
          <div ref={containerRef} className="graph-cytoscape" />
        </div>
        <Card
          className="graph-card timeline-card"
          title={t("timeline.title")}
          size="small"
        >
          {bubbleLabel && (
            <div className="timeline-bubble">
              <span className="timeline-bubble-label">
                {isDragging ? t("timeline.preview") : t("timeline.pending")}
              </span>
              <span className="timeline-bubble-range">{bubbleLabel}</span>
            </div>
          )}
          <div
            className={`timeline ${isDragging ? "is-dragging" : ""}`}
            ref={timelineRef}
            onMouseDown={handleTimelineMouseDown}
            onMouseMove={handleTimelineMouseMove}
            onMouseUp={handleTimelineMouseUp}
            onMouseLeave={handleTimelineMouseUp}
          >
            {timelineData.map((item) => {
              const isInRange =
                displayRange && item.year >= displayRange[0] && item.year <= displayRange[1];
              return (
                <div
                  key={item.year}
                  className={`timeline-item ${isInRange ? "is-selected" : ""} ${
                    isDragging ? "is-dragging" : ""
                  }`}
                  data-year={item.year}
                >
                  <div className="timeline-bar" style={{ height: `${10 + item.count * 6}px` }} />
                  <div className="timeline-year">{item.year}</div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>
      <div className="graph-detail">
        <Card className="graph-card" title={t("details.title")} size="small">
          {selected ? (
            <Space direction="vertical" size="small">
              <Title level={5}>{selected.label}</Title>
              <Text type="secondary">{selected.authors || t("details.unknown_authors")}</Text>
              <Space size="small">
                {selected.year && <Tag>{selected.year}</Tag>}
                {selected.sub_field && <Tag color="geekblue">{selected.sub_field}</Tag>}
                {selected.ccf_level && <Tag color="red">CCF {selected.ccf_level}</Tag>}
              </Space>
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
              <Paragraph ellipsis={{ rows: 6 }}>
                {selected.abstract || t("details.no_abstract")}
              </Paragraph>
              {selected.summary_one && (
                <div>
                  <Text type="secondary">{t("details.summary")}</Text>
                  <Paragraph type="secondary" ellipsis={{ rows: 2 }}>
                    {selected.summary_one}
                  </Paragraph>
                </div>
              )}
              {selected.proposed_method_name && (
                <Tag color="cyan">{selected.proposed_method_name}</Tag>
              )}
              {selected.dynamic_tags && selected.dynamic_tags.length > 0 && (
                <Space size="small" wrap>
                  {selected.dynamic_tags.map((tag) => (
                    <Tag key={`${selected.id}-${tag}`}>{tag}</Tag>
                  ))}
                </Space>
              )}
              <Space size="small">
                {selected.citation_count !== null && (
                  <Tag color="gold">
                    {t("details.citations")}: {selected.citation_count ?? 0}
                  </Tag>
                )}
                {selected.reference_count !== null && (
                  <Tag color="purple">
                    {t("details.references")}: {selected.reference_count ?? 0}
                  </Tag>
                )}
                {typeof selected.open_tasks === "number" && selected.open_tasks > 0 && (
                  <Tag color="orange">
                    {t("workspace.tasks")}: {selected.open_tasks}
                  </Tag>
                )}
                {typeof selected.experiment_count === "number" && selected.experiment_count > 0 && (
                  <Tag color="cyan">
                    {t("workspace.experiments")}: {selected.experiment_count}
                  </Tag>
                )}
              </Space>
              <Button
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
              <Button onClick={() => addToComparison(selected.id)}>{t("compare.add")}</Button>
              <Button
                onClick={() => {
                  setPathSourceId(selected.id);
                }}
              >
                {t("compare.set_source")}
              </Button>
              <Button
                onClick={() => {
                  setPathTargetId(selected.id);
                }}
              >
                {t("compare.set_target")}
              </Button>
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
              <Space size="small">
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
                <Button onClick={loadNeighbors}>{t("details.load_neighbors")}</Button>
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
              <Divider style={{ margin: "10px 0" }} />
              <Tabs
                size="small"
                items={[
                  {
                    key: "notes",
                    label: t("workspace.notes"),
                    children: (
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
                        <Input.TextArea
                          value={note?.method ?? ""}
                          onChange={(e) =>
                            setNote((prev) => ({ ...(prev ?? { paper_id: selected.id }), method: e.target.value }))
                          }
                          placeholder={t("workspace.method")}
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                        <Input.TextArea
                          value={note?.datasets ?? ""}
                          onChange={(e) =>
                            setNote((prev) => ({ ...(prev ?? { paper_id: selected.id }), datasets: e.target.value }))
                          }
                          placeholder={t("workspace.datasets")}
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                        <Input.TextArea
                          value={note?.conclusions ?? ""}
                          onChange={(e) =>
                            setNote((prev) => ({ ...(prev ?? { paper_id: selected.id }), conclusions: e.target.value }))
                          }
                          placeholder={t("workspace.conclusion")}
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                        <Input.TextArea
                          value={note?.reproducibility ?? ""}
                          onChange={(e) =>
                            setNote((prev) => ({
                              ...(prev ?? { paper_id: selected.id }),
                              reproducibility: e.target.value
                            }))
                          }
                          placeholder={t("workspace.reproducibility")}
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                        <Input.TextArea
                          value={note?.risks ?? ""}
                          onChange={(e) =>
                            setNote((prev) => ({ ...(prev ?? { paper_id: selected.id }), risks: e.target.value }))
                          }
                          placeholder={t("workspace.risks")}
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                        <Input.TextArea
                          value={note?.notes ?? ""}
                          onChange={(e) =>
                            setNote((prev) => ({ ...(prev ?? { paper_id: selected.id }), notes: e.target.value }))
                          }
                          placeholder={t("workspace.notes_free")}
                          autoSize={{ minRows: 3, maxRows: 6 }}
                        />
                        <Button
                          type="primary"
                          loading={noteSaving}
                          onClick={async () => {
                            if (!selected || !note) return;
                            setNoteSaving(true);
                            try {
                              const saved = await savePaperNotes(selected.id, {
                                method: note.method ?? null,
                                datasets: note.datasets ?? null,
                                conclusions: note.conclusions ?? null,
                                reproducibility: note.reproducibility ?? null,
                                risks: note.risks ?? null,
                                notes: note.notes ?? null
                              });
                              setNote(saved);
                              message.success(t("workspace.notes_saved"));
                            } finally {
                              setNoteSaving(false);
                            }
                          }}
                        >
                          {t("workspace.save_notes")}
                        </Button>
                      </Space>
                    )
                  },
                  {
                    key: "tasks",
                    label: (
                      <span>
                        {t("workspace.tasks")}{" "}
                        <Badge count={tasks.filter((task) => task.status !== "done").length} />
                      </span>
                    ),
                    children: (
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
                        <Input
                          value={taskTitle}
                          onChange={(e) => setTaskTitle(e.target.value)}
                          placeholder={t("workspace.task_title")}
                        />
                        <Input
                          type="date"
                          value={taskDueDate || ""}
                          onChange={(e) => setTaskDueDate(e.target.value || undefined)}
                        />
                        <Button
                          loading={taskLoading}
                          onClick={async () => {
                            if (!selected || !taskTitle.trim()) return;
                            setTaskLoading(true);
                            try {
                              await createTask({
                                paper_id: selected.id,
                                title: taskTitle.trim(),
                                due_date: taskDueDate
                              });
                              const latest = await fetchTasks({ paper_id: selected.id });
                              setTasks(latest);
                              setTaskTitle("");
                              message.success(t("workspace.task_added"));
                            } finally {
                              setTaskLoading(false);
                            }
                          }}
                        >
                          {t("workspace.add_task")}
                        </Button>
                        <List
                          size="small"
                          dataSource={tasks}
                          locale={{ emptyText: t("workspace.no_tasks") }}
                          renderItem={(item) => (
                            <List.Item
                              actions={[
                                <Button
                                  key="done"
                                  size="small"
                                  onClick={async () => {
                                    if (item.status === "done") {
                                      await updateTask(item.id, { status: "todo" });
                                    } else {
                                      await completeReview(item.id);
                                    }
                                    if (selected) {
                                      const latest = await fetchTasks({ paper_id: selected.id });
                                      setTasks(latest);
                                    }
                                  }}
                                >
                                  {item.status === "done" ? t("workspace.reopen_task") : t("workspace.finish_task")}
                                </Button>
                              ]}
                            >
                              <Space direction="vertical" size={0}>
                                <Text>{item.title}</Text>
                                <Text type="secondary">
                                  {item.due_date ? `${t("workspace.due")}: ${item.due_date}` : t("workspace.no_due")}
                                </Text>
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Space>
                    )
                  },
                  {
                    key: "experiments",
                    label: t("workspace.experiments"),
                    children: (
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
                        <Input
                          value={expName}
                          onChange={(e) => setExpName(e.target.value)}
                          placeholder={t("workspace.exp_name")}
                        />
                        <Input
                          value={expModel}
                          onChange={(e) => setExpModel(e.target.value)}
                          placeholder={t("workspace.exp_model")}
                        />
                        <Input.TextArea
                          value={expMetrics}
                          onChange={(e) => setExpMetrics(e.target.value)}
                          placeholder={t("workspace.exp_metrics")}
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                        <Space size="small" style={{ width: "100%" }} wrap>
                          <Input
                            value={expDataset}
                            onChange={(e) => setExpDataset(e.target.value)}
                            placeholder="Dataset (ACE2005)"
                            style={{ width: 150 }}
                          />
                          <Input
                            value={expSplit}
                            onChange={(e) => setExpSplit(e.target.value)}
                            placeholder="Split (test)"
                            style={{ width: 120 }}
                          />
                          <Input
                            value={expMetricName}
                            onChange={(e) => setExpMetricName(e.target.value)}
                            placeholder="Metric (F1)"
                            style={{ width: 120 }}
                          />
                          <InputNumber
                            value={expMetricValue}
                            onChange={(value) => setExpMetricValue(value ?? undefined)}
                            min={0}
                            max={100}
                            step={0.1}
                            placeholder="Value"
                            style={{ width: 110 }}
                          />
                          <Switch checked={expIsSota} onChange={setExpIsSota} />
                          <Text type="secondary">SOTA</Text>
                        </Space>
                        <Button
                          loading={experimentLoading}
                          onClick={async () => {
                            if (!selected || !expName.trim()) return;
                            setExperimentLoading(true);
                            try {
                              await createExperiment({
                                paper_id: selected.id,
                                name: expName.trim(),
                                model: expModel.trim() || undefined,
                                metrics_json: expMetrics.trim() || undefined,
                                dataset: expDataset.trim() || undefined,
                                split: expSplit.trim() || undefined,
                                metric_name: expMetricName.trim() || undefined,
                                metric_value: expMetricValue,
                                is_sota: expIsSota ? 1 : 0
                              });
                              const latest = await fetchExperiments({ paper_id: selected.id });
                              setExperiments(latest);
                              setExpName("");
                              setExpModel("");
                              setExpMetrics("");
                              setExpDataset("");
                              setExpSplit("");
                              setExpMetricName("F1");
                              setExpMetricValue(undefined);
                              setExpIsSota(false);
                              message.success(t("workspace.exp_added"));
                            } finally {
                              setExperimentLoading(false);
                            }
                          }}
                        >
                          {t("workspace.add_experiment")}
                        </Button>
                        <List
                          size="small"
                          dataSource={experiments}
                          locale={{ emptyText: t("workspace.no_experiments") }}
                          renderItem={(item) => (
                            <List.Item
                              actions={[
                                <Button
                                  key="delete"
                                  type="link"
                                  danger
                                  onClick={async () => {
                                    await deleteExperiment(item.id);
                                    if (selected) {
                                      const latest = await fetchExperiments({ paper_id: selected.id });
                                      setExperiments(latest);
                                    }
                                  }}
                                >
                                  {t("manage.delete")}
                                </Button>
                              ]}
                            >
                              <Space direction="vertical" size={0}>
                                <Text>{item.name || "-"}</Text>
                                <Text type="secondary">{item.model || "-"}</Text>
                                {item.metric_name && (
                                  <Text type="secondary">
                                    {item.dataset || item.dataset_name || "-"} / {item.split || "-"} · {item.metric_name}
                                    {item.metric_value != null ? `=${item.metric_value}` : ""}
                                    {item.is_sota ? " · SOTA" : ""}
                                  </Text>
                                )}
                                {item.metrics_json && <Text type="secondary">{item.metrics_json}</Text>}
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Space>
                    )
                  },
                  {
                    key: "comparison",
                    label: t("compare.title"),
                    children: (
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
                        <Space size="small" wrap>
                          {compareQueue.map((paperId) => (
                            <Tag
                              key={paperId}
                              closable
                              onClose={(e) => {
                                e.preventDefault();
                                removeFromComparison(paperId);
                              }}
                            >
                              #{paperId}
                            </Tag>
                          ))}
                        </Space>
                        <Button onClick={runComparison} loading={compareLoading}>
                          {t("compare.refresh")}
                        </Button>
                        <List
                          size="small"
                          loading={compareLoading}
                          dataSource={compareData?.items || []}
                          locale={{ emptyText: t("compare.empty") }}
                          renderItem={(item) => (
                            <List.Item>
                              <Space direction="vertical" size={0} style={{ width: "100%" }}>
                                <Text strong>{item.paper.title || "-"}</Text>
                                <Text type="secondary">
                                  {item.metrics
                                    .slice(0, 2)
                                    .map((m) => `${m.dataset_name || "-"} F1:${m.f1 ?? m.argument_f1 ?? m.trigger_f1 ?? "-"}`)
                                    .join(" | ")}
                                </Text>
                                <Space size="small" wrap>
                                  {item.tags.slice(0, 4).map((tag) => (
                                    <Tag key={`${item.paper.id}-${tag}`}>{tag}</Tag>
                                  ))}
                                </Space>
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Space>
                    )
                  },
                  {
                    key: "qa",
                    label: t("qa.title"),
                    children: (
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
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
                        {qaResult && (
                          <>
                            <Paragraph style={{ whiteSpace: "pre-wrap" }}>{qaResult.answer}</Paragraph>
                            <List
                              size="small"
                              header={<Text type="secondary">{t("qa.sources")}</Text>}
                              dataSource={qaResult.sources}
                              renderItem={(source) => (
                                <List.Item>
                                  <Text type="secondary">
                                    #{source.paper_id} {source.title} {source.chunk_index !== undefined ? ` [chunk ${source.chunk_index}]` : ""}
                                  </Text>
                                </List.Item>
                              )}
                            />
                          </>
                        )}
                      </Space>
                    )
                  }
                ]}
              />
            </Space>
          ) : (
            <Text type="secondary">{t("details.select_hint")}</Text>
          )}
        </Card>
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
