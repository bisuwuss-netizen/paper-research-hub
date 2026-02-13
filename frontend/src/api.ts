import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type Paper = {
  id: number;
  title?: string | null;
  authors?: string | null;
  year?: number | null;
  journal_conf?: string | null;
  ccf_level?: string | null;
  source_type?: string | null;
  sub_field?: string | null;
  read_status?: number | null;
  file_path?: string | null;
  doi?: string | null;
  abstract?: string | null;
  s2_paper_id?: string | null;
  s2_corpus_id?: string | null;
  citation_count?: number | null;
  reference_count?: number | null;
  influential_citation_count?: number | null;
  citation_velocity?: number | null;
  url?: string | null;
  venue?: string | null;
  keywords?: string | null;
  cluster_id?: string | null;
  zotero_item_key?: string | null;
  zotero_library?: string | null;
  zotero_item_id?: number | null;
  summary_one?: string | null;
  proposed_method_name?: string | null;
  dynamic_tags?: string | null;
  embedding?: string | null;
  open_sub_field?: string | null;
  created_at?: number | null;
  updated_at?: number | null;
};

export type GraphNode = {
  id: number;
  label: string;
  size: number;
  color: string;
  year?: number | null;
  authors?: string | null;
  abstract?: string | null;
  sub_field?: string | null;
  read_status?: number | null;
  ccf_level?: string | null;
  source_type?: string | null;
  citation_count?: number | null;
  reference_count?: number | null;
  pagerank?: number | null;
  citation_velocity?: number | null;
  doi?: string | null;
  url?: string | null;
  zotero_item_key?: string | null;
  zotero_library?: string | null;
  zotero_item_id?: number | null;
  summary_one?: string | null;
  proposed_method_name?: string | null;
  dynamic_tags?: string[] | null;
  open_sub_field?: string | null;
  open_tasks?: number | null;
  overdue_tasks?: number | null;
  experiment_count?: number | null;
};

export type GraphEdge = {
  id: string;
  source: number;
  target: number;
  confidence?: number;
  edge_source?: string | null;
  intent?: string | null;
  intent_confidence?: number | null;
  context_snippet?: string | null;
  intent_source?: string | null;
};

export type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: {
    sub_fields: string[];
    year_min: number | null;
    year_max: number | null;
    year_counts: { year: number; count: number }[];
    year_min_all?: number | null;
    year_max_all?: number | null;
    year_counts_all?: { year: number; count: number }[];
    total_nodes?: number;
    limit?: number | null;
    offset?: number | null;
    size_by?: string;
  };
};

export type GraphMiniResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type Subfield = {
  id: number;
  name: string;
  description?: string | null;
  active?: number;
};

export type SyncStatus = Record<string, number>;

export type SyncStatusResponse = SyncStatus & {
  jobs?: Record<string, number>;
  alerts_open?: number;
};

export type PaperNote = {
  paper_id: number;
  method?: string | null;
  datasets?: string | null;
  conclusions?: string | null;
  reproducibility?: string | null;
  risks?: string | null;
  notes?: string | null;
  created_at?: number | null;
  updated_at?: number | null;
};

export type PaperSchema = {
  paper_id: number;
  event_types: Array<{
    name: string;
    aliases?: string[];
    roles?: string[];
  }>;
  role_types: string[];
  schema_notes?: string | null;
  confidence?: number | null;
  source?: string | null;
  updated_at?: number | null;
};

export type BacklinksResponse = {
  paper_id: number;
  count: number;
  items: Array<{
    source_paper_id: number;
    target_paper_id: number;
    link_text: string;
    updated_at?: number | null;
    source_title?: string | null;
    source_year?: number | null;
    source_sub_field?: string | null;
  }>;
};

export type ReadingTask = {
  id: number;
  paper_id: number;
  title: string;
  status: string;
  priority: number;
  due_date?: string | null;
  next_review_at?: number | null;
  interval_days?: number | null;
  last_review_at?: number | null;
};

export type Experiment = {
  id: number;
  paper_id: number;
  name?: string | null;
  model?: string | null;
  params_json?: string | null;
  metrics_json?: string | null;
  result_summary?: string | null;
  artifact_path?: string | null;
  dataset_name?: string | null;
  trigger_f1?: number | null;
  argument_f1?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  dataset?: string | null;
  split?: string | null;
  metric_name?: string | null;
  metric_value?: number | null;
  is_sota?: number | null;
  created_at?: number | null;
  updated_at?: number | null;
};

export type SearchResult = {
  paper: Paper;
  score: number;
  bm25_score: number;
  semantic_score: number;
  chunk_score?: number;
  chunk_index?: number | null;
  snippet?: string;
};

export type TopicEvolution = {
  trends: Array<{
    year: number;
    total: number;
    sub_fields: Record<string, number>;
  }>;
  hotspots: Array<{
    year: number;
    papers: Array<{
      id: number;
      title?: string;
      sub_field?: string;
      velocity: number;
      citation_count: number;
    }>;
  }>;
  bursts: Array<{
    sub_field: string;
    year: number;
    count: number;
    growth: number;
    growth_ratio: number;
  }>;
};

export type TopicRiver = {
  river: Array<{
    year: number;
    sub_field: string;
    count: number;
    ratio: number;
  }>;
  years: number[];
  sub_fields: string[];
  bursts: TopicEvolution["bursts"];
};

export type ChatWithPapersResponse = {
  query: string;
  paper_ids: number[];
  answer: string;
  sources: Array<{
    paper_id?: number;
    title?: string;
    score?: number;
    chunk_index?: number;
  }>;
  context_count: number;
  trace_score?: number;
  session_id?: number | null;
  routes?: string[];
};

export type SotaBoard = {
  dataset?: string | null;
  items: Array<{
    id: number;
    paper_id: number;
    dataset_name?: string | null;
    precision?: number | null;
    recall?: number | null;
    f1?: number | null;
    trigger_f1?: number | null;
    argument_f1?: number | null;
    source?: string | null;
    confidence?: number | null;
    title?: string | null;
    year?: number | null;
    sub_field?: string | null;
    ccf_level?: string | null;
    doi?: string | null;
    url?: string | null;
  }>;
  datasets: string[];
  count: number;
};

export type MetricLeaderboard = {
  dataset?: string | null;
  metric: string;
  top_items: Array<{
    paper_id: number;
    dataset_name?: string | null;
    metric_value?: number | null;
    table_id?: string | null;
    row_index?: number | null;
    col_index?: number | null;
    cell_text?: string | null;
    provenance?: Record<string, any>;
    source?: string | null;
    confidence?: number | null;
    title?: string | null;
    year?: number | null;
    sub_field?: string | null;
    ccf_level?: string | null;
  }>;
  trend: Array<{
    year: number;
    best_value?: number | null;
    avg_value?: number | null;
    sample_count: number;
  }>;
};

export type MetricProvenance = {
  items: Array<{
    id: number;
    paper_id: number;
    metric_key?: string | null;
    metric_value?: number | null;
    dataset_name?: string | null;
    table_id?: string | null;
    row_index?: number | null;
    col_index?: number | null;
    cell_text?: string | null;
    parser?: string | null;
    confidence?: number | null;
    created_at?: number | null;
  }>;
  count: number;
};

export type ChatSession = {
  id: number;
  title?: string | null;
  language?: string | null;
  created_at?: number | null;
  updated_at?: number | null;
};

export type ChatMessage = {
  id: number;
  session_id: number;
  role: string;
  content: string;
  paper_ids?: number[];
  trace_score?: number | null;
  sources?: Array<Record<string, any>>;
  created_at?: number | null;
};

export type IdeaCapsule = {
  id: number;
  title: string;
  content: string;
  status: "seed" | "incubating" | "validated" | "archived" | string;
  priority: number;
  linked_papers: number[];
  tags: string[];
  source_note_paper_id?: number | null;
  created_at?: number | null;
  updated_at?: number | null;
};

export type JobRun = {
  id: number;
  job_type: string;
  status: string;
  payload?: Record<string, any>;
  result?: Record<string, any>;
  attempts?: number;
  error?: string | null;
  started_at?: number | null;
  finished_at?: number | null;
  next_retry_at?: number | null;
  created_at?: number | null;
  updated_at?: number | null;
};

export type JobAlert = {
  id: number;
  level?: string;
  source_job_id?: number | null;
  message?: string | null;
  payload?: Record<string, any>;
  resolved?: number;
  created_at?: number | null;
  resolved_at?: number | null;
};

export type OpenTagDiscovery = {
  candidates: string[];
  clusters: Array<{
    label: string;
    members: string[];
    count: number;
  }>;
  added: number;
};

export type ShortestPathResponse = {
  source_id: number;
  target_id: number;
  path: {
    nodes: number[];
    edges: Array<[number, number]>;
    distance: number | null;
  };
  papers: Array<{
    id: number;
    title?: string | null;
    year?: number | null;
    sub_field?: string | null;
  }>;
};

export type ComparePapersResponse = {
  items: Array<{
    paper: Paper;
    metrics: Array<{
      paper_id: number;
      dataset_name?: string | null;
      precision?: number | null;
      recall?: number | null;
      f1?: number | null;
      trigger_f1?: number | null;
      argument_f1?: number | null;
      confidence?: number | null;
    }>;
    tags: string[];
  }>;
  count: number;
};

export type Report = {
  id: number;
  period_type: "weekly" | "monthly";
  period_start: string;
  period_end: string;
  payload_json?: string;
  payload?: Record<string, any>;
  created_at: number;
};

export type ZoteroMappingTemplate = {
  id?: number;
  name: string;
  mapping: Record<string, string>;
  mapping_json?: string;
  created_at?: number;
  updated_at?: number;
};

export type ZoteroSyncLog = {
  id: number;
  paper_id?: number | null;
  direction?: string | null;
  action?: string | null;
  status?: string | null;
  conflict_strategy?: string | null;
  details?: Record<string, any>;
  created_at?: number | null;
};

export async function fetchNeighbors(params: {
  node_id: number;
  depth?: number;
  limit?: number;
}): Promise<GraphMiniResponse> {
  const res = await axios.get(`${API_BASE}/api/graph/neighbors`, { params });
  return res.data;
}

export async function fetchSubfields(active_only = true): Promise<Subfield[]> {
  const res = await axios.get(`${API_BASE}/api/subfields`, { params: { active_only } });
  return res.data;
}

export async function createSubfield(payload: {
  name: string;
  description?: string;
  active?: number;
}): Promise<Subfield> {
  const res = await axios.post(`${API_BASE}/api/subfields`, payload);
  return res.data;
}

export async function updateSubfield(
  id: number,
  payload: Partial<{ name: string; description: string; active: number }>
): Promise<Subfield> {
  const res = await axios.patch(`${API_BASE}/api/subfields/${id}`, payload);
  return res.data;
}

export async function deleteSubfield(id: number): Promise<void> {
  await axios.delete(`${API_BASE}/api/subfields/${id}`);
}

export type RecommendationResponse = {
  strategy: "foundation" | "sota" | "cluster";
  highlight_nodes?: number[];
  clusters?: Record<string, string>;
};

export type PathExportResponse = {
  strategy: "foundation" | "sota";
  path: {
    id: number;
    title?: string | null;
    authors?: string | null;
    year?: number | null;
    venue?: string | null;
    doi?: string | null;
    url?: string | null;
  }[];
};

export async function fetchPapers(): Promise<Paper[]> {
  const res = await axios.get(`${API_BASE}/api/papers`);
  return res.data;
}

export async function uploadPaper(file: File): Promise<Paper> {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${API_BASE}/api/papers/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return res.data;
}

export async function fetchGraph(params?: {
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
}): Promise<GraphResponse> {
  const res = await axios.get(`${API_BASE}/api/graph`, { params });
  return res.data;
}

export async function fetchRecommendations(params: {
  strategy: "foundation" | "sota" | "cluster";
  sub_field?: string;
  year_from?: number;
  year_to?: number;
}): Promise<RecommendationResponse> {
  const res = await axios.get(`${API_BASE}/api/recommendations`, { params });
  return res.data;
}

export async function exportPath(params: {
  strategy: "foundation" | "sota";
  sub_field?: string;
  year_from?: number;
  year_to?: number;
}): Promise<PathExportResponse> {
  const res = await axios.get(`${API_BASE}/api/recommendations/path`, { params });
  return res.data;
}

export async function updatePaper(
  id: number,
  payload: Partial<
    Pick<
      Paper,
      "read_status" | "zotero_item_key" | "zotero_library" | "zotero_item_id" | "sub_field"
    >
  >
): Promise<Paper> {
  const res = await axios.patch(`${API_BASE}/api/papers/${id}`, payload);
  return res.data;
}

export async function matchZotero(id: number): Promise<Paper> {
  const res = await axios.post(`${API_BASE}/api/papers/${id}/zotero-match`);
  return res.data;
}

export async function pushZotero(id: number): Promise<void> {
  await axios.post(`${API_BASE}/api/zotero/push/${id}`);
}

export async function zoteroMatchAll(limit = 20): Promise<{ matched: number }> {
  const res = await axios.post(`${API_BASE}/api/zotero/match-all`, null, { params: { limit } });
  return res.data;
}

export async function zoteroPushAll(limit = 20): Promise<{ updated: number }> {
  const res = await axios.post(`${API_BASE}/api/zotero/push-all`, null, { params: { limit } });
  return res.data;
}

export async function zoteroSyncIds(limit = 50): Promise<{ synced: number }> {
  const res = await axios.post(`${API_BASE}/api/zotero/sync-ids`, null, { params: { limit } });
  return res.data;
}

export async function fetchSyncStatus(): Promise<SyncStatusResponse> {
  const res = await axios.get(`${API_BASE}/api/sync/status`);
  return res.data;
}

export async function enqueueAllSync(): Promise<{ enqueued: number }> {
  const res = await axios.post(`${API_BASE}/api/sync/enqueue-all`);
  return res.data;
}

export async function runSync(limit = 5): Promise<{ processed: number; failed: number }> {
  const res = await axios.post(`${API_BASE}/api/sync/run`, null, { params: { limit } });
  return res.data;
}

export async function cleanupCitations(): Promise<{ removed: number }> {
  const res = await axios.post(`${API_BASE}/api/maintenance/cleanup`);
  return res.data;
}

export async function backfillPapers(params?: {
  limit?: number;
  summary?: boolean;
  references?: boolean;
  embeddings?: boolean;
  metrics?: boolean;
  force?: boolean;
}): Promise<{
  processed: number;
  summary_added: number;
  references_parsed: number;
  chunks_indexed?: number;
  metrics_upserted?: number;
  metric_cells_upserted?: number;
  auto_experiments?: number;
  schemas_extracted?: number;
}> {
  const res = await axios.post(`${API_BASE}/api/papers/backfill`, null, { params });
  return res.data;
}

export async function searchPapers(
  q: string,
  top_k = 20
): Promise<{ query: string; results: SearchResult[]; fallback?: boolean }> {
  const res = await axios.get(`${API_BASE}/api/search`, { params: { q, top_k } });
  return res.data;
}

export async function fetchPaperNotes(paperId: number): Promise<PaperNote> {
  const res = await axios.get(`${API_BASE}/api/papers/${paperId}/notes`);
  return res.data;
}

export async function savePaperNotes(
  paperId: number,
  payload: Partial<Omit<PaperNote, "paper_id">>
): Promise<PaperNote> {
  const res = await axios.put(`${API_BASE}/api/papers/${paperId}/notes`, payload);
  return res.data;
}

export async function fetchPaperBacklinks(paperId: number): Promise<BacklinksResponse> {
  const res = await axios.get(`${API_BASE}/api/papers/${paperId}/backlinks`);
  return res.data;
}

export async function fetchPaperSchema(paperId: number): Promise<PaperSchema> {
  const res = await axios.get(`${API_BASE}/api/papers/${paperId}/schema`);
  return res.data;
}

export async function extractPaperSchema(
  paperId: number,
  force = false
): Promise<PaperSchema> {
  const res = await axios.post(`${API_BASE}/api/papers/${paperId}/schema/extract`, null, {
    params: { force }
  });
  return res.data;
}

export async function searchSchemas(
  keyword: string,
  limit = 50
): Promise<{
  keyword: string;
  count: number;
  items: Array<{
    paper_id: number;
    title?: string;
    year?: number;
    sub_field?: string;
    event_types: Array<{ name: string; roles?: string[] }>;
    role_types: string[];
  }>;
}> {
  const res = await axios.get(`${API_BASE}/api/schemas/search`, {
    params: { keyword, limit }
  });
  return res.data;
}

export async function fetchTasks(params?: {
  paper_id?: number;
  status?: string;
}): Promise<ReadingTask[]> {
  const res = await axios.get(`${API_BASE}/api/tasks`, { params });
  return res.data;
}

export async function createTask(payload: {
  paper_id: number;
  title: string;
  priority?: number;
  due_date?: string;
  next_review_at?: number;
  interval_days?: number;
}): Promise<ReadingTask> {
  const res = await axios.post(`${API_BASE}/api/tasks`, payload);
  return res.data;
}

export async function updateTask(
  taskId: number,
  payload: Partial<Pick<ReadingTask, "title" | "status" | "priority" | "due_date" | "next_review_at" | "interval_days">>
): Promise<ReadingTask> {
  const res = await axios.patch(`${API_BASE}/api/tasks/${taskId}`, payload);
  return res.data;
}

export async function completeReview(taskId: number): Promise<ReadingTask> {
  const res = await axios.post(`${API_BASE}/api/tasks/${taskId}/complete-review`);
  return res.data;
}

export async function fetchExperiments(params?: { paper_id?: number }): Promise<Experiment[]> {
  const res = await axios.get(`${API_BASE}/api/experiments`, { params });
  return res.data;
}

export async function createExperiment(payload: {
  paper_id: number;
  name?: string;
  model?: string;
  params_json?: string;
  metrics_json?: string;
  result_summary?: string;
  artifact_path?: string;
  dataset_name?: string;
  trigger_f1?: number;
  argument_f1?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  dataset?: string;
  split?: string;
  metric_name?: string;
  metric_value?: number;
  is_sota?: number;
}): Promise<Experiment> {
  const res = await axios.post(`${API_BASE}/api/experiments`, payload);
  return res.data;
}

export async function updateExperiment(
  experimentId: number,
  payload: Partial<
    Pick<
      Experiment,
      "name" | "model" | "params_json" | "metrics_json" | "result_summary" | "artifact_path" |
      "dataset_name" | "trigger_f1" | "argument_f1" | "precision" | "recall" | "f1" |
      "dataset" | "split" | "metric_name" | "metric_value" | "is_sota"
    >
  >
): Promise<Experiment> {
  const res = await axios.patch(`${API_BASE}/api/experiments/${experimentId}`, payload);
  return res.data;
}

export async function deleteExperiment(experimentId: number): Promise<void> {
  await axios.delete(`${API_BASE}/api/experiments/${experimentId}`);
}

export async function fetchDuplicateGroups(params?: {
  title_threshold?: number;
}): Promise<{ groups: Array<{ type: string; key: string; paper_ids: number[]; confidence: number }>; count: number }> {
  const res = await axios.get(`${API_BASE}/api/quality/duplicates`, { params });
  return res.data;
}

export async function fetchConflicts(): Promise<{
  conflicts: Array<{
    doi: string;
    paper_ids: number[];
    title_variants: string[];
    year_variants: number[];
  }>;
  count: number;
}> {
  const res = await axios.get(`${API_BASE}/api/quality/conflicts`);
  return res.data;
}

export async function mergePapers(payload: {
  source_paper_id: number;
  target_paper_id: number;
}): Promise<{ status: string; source: number; target: number }> {
  const res = await axios.post(`${API_BASE}/api/quality/merge`, payload);
  return res.data;
}

export async function autoMerge(params?: {
  limit?: number;
  title_threshold?: number;
}): Promise<{ merged: number }> {
  const res = await axios.post(`${API_BASE}/api/quality/auto-merge`, null, { params });
  return res.data;
}

export async function fetchTopicEvolution(): Promise<TopicEvolution> {
  const res = await axios.get(`${API_BASE}/api/analytics/topic-evolution`);
  return res.data;
}

export async function fetchTopicRiver(): Promise<TopicRiver> {
  const res = await axios.get(`${API_BASE}/api/analytics/topic-river`);
  return res.data;
}

export async function chatWithPapers(payload: {
  query: string;
  paper_ids?: number[];
  top_k?: number;
  language?: "zh" | "en";
  session_id?: number;
  use_memory?: boolean;
}): Promise<ChatWithPapersResponse> {
  const res = await axios.post(`${API_BASE}/api/chat/papers`, payload);
  return res.data;
}

export async function streamChatWithPapers(
  payload: {
    query: string;
    paper_ids?: number[];
    top_k?: number;
    language?: "zh" | "en";
    session_id?: number;
    use_memory?: boolean;
  },
  handlers: {
    onMeta?: (meta: {
      query: string;
      paper_ids: number[];
      context_count: number;
      session_id?: number;
      routes?: string[];
    }) => void;
    onDelta?: (delta: string) => void;
    onSources?: (sources: ChatWithPapersResponse["sources"], traceScore?: number) => void;
    onDone?: () => void;
  }
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/papers/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const line = chunk
        .split("\n")
        .find((entry) => entry.trim().startsWith("data:"));
      if (!line) continue;
      const payloadText = line.replace(/^data:\s*/, "");
      if (!payloadText) continue;
      let event: any;
      try {
        event = JSON.parse(payloadText);
      } catch {
        continue;
      }
      if (event.type === "meta") {
        handlers.onMeta?.({
          query: event.query || "",
          paper_ids: event.paper_ids || [],
          context_count: event.context_count || 0,
          session_id: event.session_id,
          routes: event.routes || []
        });
      } else if (event.type === "delta") {
        handlers.onDelta?.(event.delta || "");
      } else if (event.type === "sources") {
        handlers.onSources?.(event.sources || [], event.trace_score);
      } else if (event.type === "done") {
        handlers.onDone?.();
      }
    }
  }
}

export async function fetchSotaBoard(params?: {
  dataset?: string;
  limit?: number;
}): Promise<SotaBoard> {
  const res = await axios.get(`${API_BASE}/api/metrics/sota`, { params });
  return res.data;
}

export async function fetchMetricLeaderboard(params?: {
  dataset?: string;
  metric?: "f1" | "precision" | "recall" | "trigger_f1" | "argument_f1";
  limit?: number;
}): Promise<MetricLeaderboard> {
  const res = await axios.get(`${API_BASE}/api/metrics/leaderboard`, { params });
  return res.data;
}

export async function fetchMetricProvenance(params?: {
  paper_id?: number;
  dataset?: string;
  limit?: number;
}): Promise<MetricProvenance> {
  const res = await axios.get(`${API_BASE}/api/metrics/provenance`, { params });
  return res.data;
}

export async function discoverOpenTags(params?: {
  limit?: number;
  add_to_subfields?: boolean;
}): Promise<OpenTagDiscovery> {
  const res = await axios.get(`${API_BASE}/api/subfields/discover-open-tags`, { params });
  return res.data;
}

export async function classifyCitationIntent(payload: {
  edge_ids?: string[];
  limit?: number;
}): Promise<{
  updated: number;
  items: Array<{
    id: string;
    source: number;
    target: number;
    intent: string;
    intent_confidence: number;
  }>;
}> {
  const res = await axios.post(`${API_BASE}/api/citations/intent/classify`, payload);
  return res.data;
}

export async function fetchShortestPath(params: {
  source_id: number;
  target_id: number;
  direction?: "any" | "out" | "in";
}): Promise<ShortestPathResponse> {
  const res = await axios.get(`${API_BASE}/api/graph/shortest-path`, { params });
  return res.data;
}

export async function comparePapers(payload: { paper_ids: number[] }): Promise<ComparePapersResponse> {
  const res = await axios.post(`${API_BASE}/api/papers/compare`, payload);
  return res.data;
}

export async function generateReport(period: "weekly" | "monthly"): Promise<Report> {
  const res = await axios.post(`${API_BASE}/api/reports/generate`, null, { params: { period } });
  return res.data;
}

export async function fetchReports(params?: {
  period?: "weekly" | "monthly";
  limit?: number;
}): Promise<Report[]> {
  const res = await axios.get(`${API_BASE}/api/reports`, { params });
  return res.data;
}

export async function fetchZoteroTemplate(name = "default"): Promise<ZoteroMappingTemplate> {
  const res = await axios.get(`${API_BASE}/api/zotero/mapping-template`, { params: { name } });
  return res.data;
}

export async function saveZoteroTemplate(payload: {
  name: string;
  mapping: Record<string, string>;
}): Promise<ZoteroMappingTemplate> {
  const res = await axios.put(`${API_BASE}/api/zotero/mapping-template`, payload);
  return res.data;
}

export async function fetchZoteroLogs(limit = 100): Promise<ZoteroSyncLog[]> {
  const res = await axios.get(`${API_BASE}/api/zotero/sync-logs`, { params: { limit } });
  return res.data;
}

export async function syncZoteroIncremental(payload: {
  direction: "both" | "pull" | "push";
  conflict_strategy: "prefer_local" | "prefer_zotero" | "manual";
  limit?: number;
}): Promise<{ synced: number; conflicts: number }> {
  const res = await axios.post(`${API_BASE}/api/zotero/sync-incremental`, payload);
  return res.data;
}

export async function createChatSession(payload?: {
  title?: string;
  language?: "zh" | "en";
}): Promise<ChatSession> {
  const res = await axios.post(`${API_BASE}/api/chat/sessions`, payload || {});
  return res.data;
}

export async function fetchChatSessions(limit = 20): Promise<ChatSession[]> {
  const res = await axios.get(`${API_BASE}/api/chat/sessions`, { params: { limit } });
  return res.data;
}

export async function fetchChatMessages(sessionId: number, limit = 100): Promise<ChatMessage[]> {
  const res = await axios.get(`${API_BASE}/api/chat/sessions/${sessionId}/messages`, { params: { limit } });
  return res.data;
}

export async function fetchIdeaCapsules(params?: {
  status?: string;
  paper_id?: number;
  limit?: number;
}): Promise<{ items: IdeaCapsule[]; count: number }> {
  const res = await axios.get(`${API_BASE}/api/idea-capsules`, { params });
  return res.data;
}

export async function fetchIdeaCapsuleBoard(): Promise<{
  board: Record<string, IdeaCapsule[]>;
  summary: Record<string, number>;
}> {
  const res = await axios.get(`${API_BASE}/api/idea-capsules/board`);
  return res.data;
}

export async function createIdeaCapsule(payload: {
  title: string;
  content: string;
  status?: string;
  priority?: number;
  linked_papers?: number[];
  tags?: string[];
  source_note_paper_id?: number;
}): Promise<IdeaCapsule> {
  const res = await axios.post(`${API_BASE}/api/idea-capsules`, payload);
  return res.data;
}

export async function updateIdeaCapsule(
  capsuleId: number,
  payload: Partial<{
    title: string;
    content: string;
    status: string;
    priority: number;
    linked_papers: number[];
    tags: string[];
  }>
): Promise<IdeaCapsule> {
  const res = await axios.patch(`${API_BASE}/api/idea-capsules/${capsuleId}`, payload);
  return res.data;
}

export async function deleteIdeaCapsule(capsuleId: number): Promise<void> {
  await axios.delete(`${API_BASE}/api/idea-capsules/${capsuleId}`);
}

export async function fetchJobHistory(params?: {
  status?: string;
  limit?: number;
}): Promise<{ items: JobRun[]; count: number }> {
  const res = await axios.get(`${API_BASE}/api/jobs/history`, { params });
  return res.data;
}

export async function retryJob(jobId: number): Promise<{ job_id: number; status: string; payload?: Record<string, any> }> {
  const res = await axios.post(`${API_BASE}/api/jobs/${jobId}/retry`);
  return res.data;
}

export async function runQueuedJobs(limit = 5): Promise<{ executed: number; failed: number }> {
  const res = await axios.post(`${API_BASE}/api/jobs/run-queued`, null, { params: { limit } });
  return res.data;
}

export async function fetchJobAlerts(params?: {
  resolved?: boolean;
  limit?: number;
}): Promise<{ items: JobAlert[]; count: number }> {
  const res = await axios.get(`${API_BASE}/api/jobs/alerts`, { params });
  return res.data;
}

export async function resolveJobAlert(alertId: number): Promise<{ status: string; id: number }> {
  const res = await axios.post(`${API_BASE}/api/jobs/alerts/${alertId}/resolve`);
  return res.data;
}

export async function fetchSchemaOntology(min_support = 2): Promise<{
  nodes: Array<{ id: number; name: string; type: string; aliases: string[]; paper_count: number }>;
  edges: Array<{ source: number; target: number; co_count: number }>;
  count: number;
}> {
  const res = await axios.get(`${API_BASE}/api/schemas/ontology`, { params: { min_support } });
  return res.data;
}

export async function alignSchemaOntology(limit = 500): Promise<{
  papers_processed: number;
  aligned_links: number;
  concepts: number;
}> {
  const res = await axios.post(`${API_BASE}/api/schemas/align`, null, { params: { limit } });
  return res.data;
}
