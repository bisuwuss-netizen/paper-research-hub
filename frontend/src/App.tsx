import { useEffect, useMemo, useState } from "react";
import {
  Layout,
  Typography,
  Upload,
  Table,
  Tag,
  message,
  Space,
  Button,
  Tabs,
  ConfigProvider,
  Input,
  List,
  Card,
  Select,
  InputNumber
} from "antd";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import type { UploadRequestOption as RcCustomRequestOptions } from "rc-upload/lib/interface";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  autoMerge,
  completeReview,
  fetchConflicts,
  fetchDuplicateGroups,
  fetchPapers,
  fetchReports,
  fetchSotaBoard,
  fetchTasks,
  fetchTopicEvolution,
  fetchTopicRiver,
  fetchZoteroLogs,
  fetchZoteroTemplate,
  generateReport,
  mergePapers,
  saveZoteroTemplate,
  searchPapers,
  syncZoteroIncremental,
  updateTask,
  uploadPaper,
  type Paper,
  type ReadingTask,
  type Report,
  type SearchResult,
  type SotaBoard,
  type TopicEvolution,
  type TopicRiver,
  type ZoteroSyncLog
} from "./api";
import GraphView from "./GraphView";
import { createT, type Lang } from "./i18n";
import "./styles.css";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const { Header, Content } = Layout;

function DashboardPage({
  t,
  lang,
  papers,
  loading,
  uploading,
  loadPapers,
  onUpload
}: {
  t: (key: string) => string;
  lang: Lang;
  papers: Paper[];
  loading: boolean;
  uploading: boolean;
  loadPapers: () => Promise<void>;
  onUpload: (options: RcCustomRequestOptions) => Promise<void>;
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchFallback, setSearchFallback] = useState(false);

  const [tasks, setTasks] = useState<ReadingTask[]>([]);
  const [taskLoading, setTaskLoading] = useState(false);

  const [topicEvolution, setTopicEvolution] = useState<TopicEvolution | null>(null);
  const [topicRiver, setTopicRiver] = useState<TopicRiver | null>(null);
  const [sotaBoard, setSotaBoard] = useState<SotaBoard | null>(null);
  const [sotaDataset, setSotaDataset] = useState<string | undefined>();
  const [reports, setReports] = useState<Report[]>([]);
  const [insightLoading, setInsightLoading] = useState(false);

  const [duplicates, setDuplicates] = useState<Array<{ type: string; key: string; paper_ids: number[]; confidence: number }>>([]);
  const [conflicts, setConflicts] = useState<
    Array<{ doi: string; paper_ids: number[]; title_variants: string[]; year_variants: number[] }>
  >([]);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [mergeSourceId, setMergeSourceId] = useState<number | undefined>();
  const [mergeTargetId, setMergeTargetId] = useState<number | undefined>();

  const [zoteroTemplate, setZoteroTemplate] = useState(
    '{"title":"title","abstract":"abstractNote","authors":"creators","doi":"DOI","venue":"publicationTitle","year":"date","summary_one":"extra"}'
  );
  const [zoteroDirection, setZoteroDirection] = useState<"both" | "pull" | "push">("both");
  const [zoteroStrategy, setZoteroStrategy] = useState<"prefer_local" | "prefer_zotero" | "manual">("prefer_local");
  const [zoteroLimit, setZoteroLimit] = useState(20);
  const [zoteroLogs, setZoteroLogs] = useState<ZoteroSyncLog[]>([]);
  const [zoteroLoading, setZoteroLoading] = useState(false);

  const columns = useMemo(
    () => [
      {
        title: t("table.title"),
        dataIndex: "title",
        key: "title",
        render: (value: string | null) => value || t("table.untitled")
      },
      {
        title: t("table.authors"),
        dataIndex: "authors",
        key: "authors",
        render: (value: string | null) => value || "-"
      },
      {
        title: t("table.year"),
        dataIndex: "year",
        key: "year",
        width: 100,
        render: (value: number | null) => value || "-"
      },
      {
        title: t("table.subfield"),
        dataIndex: "sub_field",
        key: "sub_field",
        render: (value: string | null) => (value ? <Tag>{value}</Tag> : "-")
      },
      {
        title: t("table.abstract"),
        dataIndex: "abstract",
        key: "abstract",
        render: (value: string | null) => (value ? value.slice(0, 120) + "..." : "-")
      }
    ],
    [t]
  );

  const taskStats = useMemo(() => {
    const overdue = tasks.filter((task) => task.status !== "done" && task.due_date && task.due_date < new Date().toISOString().slice(0, 10)).length;
    const reviewDue = tasks.filter((task) => task.status !== "done" && task.next_review_at && task.next_review_at * 1000 < Date.now()).length;
    const done = tasks.filter((task) => task.status === "done").length;
    return { overdue, reviewDue, done };
  }, [tasks]);

  const riverData = useMemo(() => {
    if (!topicRiver) return [];
    const topSubFields = topicRiver.sub_fields.slice(0, 8);
    const yearMap = new Map<number, Record<string, number>>();
    topicRiver.years.forEach((year) => {
      yearMap.set(year, { year } as unknown as Record<string, number>);
    });
    topicRiver.river.forEach((item) => {
      if (!topSubFields.includes(item.sub_field)) return;
      const row = yearMap.get(item.year) || ({ year: item.year } as unknown as Record<string, number>);
      row[item.sub_field] = item.count;
      yearMap.set(item.year, row);
    });
    return Array.from(yearMap.values()).sort((a, b) => Number(a.year) - Number(b.year));
  }, [topicRiver]);

  const refreshTasks = async () => {
    setTaskLoading(true);
    try {
      const rows = await fetchTasks();
      setTasks(rows);
    } finally {
      setTaskLoading(false);
    }
  };

  const refreshInsights = async () => {
    setInsightLoading(true);
    try {
      const [evolution, river, reportRows, sota] = await Promise.all([
        fetchTopicEvolution(),
        fetchTopicRiver(),
        fetchReports({ limit: 20 }),
        fetchSotaBoard({ dataset: sotaDataset, limit: 50 })
      ]);
      setTopicEvolution(evolution);
      setTopicRiver(river);
      setReports(reportRows);
      setSotaBoard(sota);
    } finally {
      setInsightLoading(false);
    }
  };

  const refreshQuality = async () => {
    setQualityLoading(true);
    try {
      const [dup, conf] = await Promise.all([fetchDuplicateGroups(), fetchConflicts()]);
      setDuplicates(dup.groups);
      setConflicts(conf.conflicts);
    } finally {
      setQualityLoading(false);
    }
  };

  const refreshZotero = async () => {
    setZoteroLoading(true);
    try {
      const [template, logs] = await Promise.all([fetchZoteroTemplate(), fetchZoteroLogs(30)]);
      setZoteroTemplate(JSON.stringify(template.mapping || {}, null, 2));
      setZoteroLogs(logs);
    } catch {
      // ignore
    } finally {
      setZoteroLoading(false);
    }
  };

  useEffect(() => {
    refreshTasks();
    refreshInsights();
    refreshQuality();
    refreshZotero();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    refreshInsights();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sotaDataset]);

  return (
    <>
      <div className="hero-card">
        <div>
          <Typography.Title level={4} className="hero-title">
            {t("app.subtitle")}
          </Typography.Title>
          <Typography.Text className="hero-copy">
            {lang === "zh"
              ? "上传 PDF、构建引用网络、做语义检索并管理阅读任务。"
              : "Upload PDFs, build citation graphs, run semantic search, and manage reading tasks."}
          </Typography.Text>
        </div>
        <div className="hero-actions">
          <Button type="primary" onClick={loadPapers} loading={loading}>
            {t("btn.refresh")}
          </Button>
        </div>
      </div>

      <Tabs
        className="app-tabs"
        items={[
          {
            key: "list",
            label: t("tab.list"),
            children: (
              <>
                <div className="card">
                  <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                    <Upload.Dragger
                      name="file"
                      multiple={false}
                      accept=".pdf"
                      customRequest={onUpload}
                      showUploadList={false}
                      disabled={uploading}
                    >
                      <p className="ant-upload-drag-icon">PDF</p>
                      <p className="ant-upload-text">{t("upload.text")}</p>
                      <p className="ant-upload-hint">{t("upload.hint")}</p>
                    </Upload.Dragger>
                  </Space>
                </div>
                <div className="card">
                  <Table
                    rowKey="id"
                    loading={loading}
                    columns={columns}
                    dataSource={papers}
                    pagination={{ pageSize: 8 }}
                  />
                </div>
              </>
            )
          },
          {
            key: "search",
            label: t("tab.search"),
            children: (
              <div className="card">
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Input.Search
                    enterButton={t("search.run")}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder={t("search.placeholder")}
                    loading={searching}
                    onSearch={async (value) => {
                      if (!value.trim()) return;
                      setSearching(true);
                      try {
                        const data = await searchPapers(value, 20);
                        setSearchResults(data.results);
                        setSearchFallback(Boolean(data.fallback));
                      } finally {
                        setSearching(false);
                      }
                    }}
                  />
                  {searchFallback && <Typography.Text type="secondary">{t("search.fallback")}</Typography.Text>}
                  <List
                    dataSource={searchResults}
                    locale={{ emptyText: t("search.empty") }}
                    renderItem={(item) => (
                      <List.Item>
                        <Space direction="vertical" style={{ width: "100%" }}>
                          <Typography.Text strong>{item.paper.title || "-"}</Typography.Text>
                          <Space size="small" wrap>
                            <Tag>{t("search.score")}: {item.score.toFixed(3)}</Tag>
                            <Tag>{t("search.bm25")}: {item.bm25_score.toFixed(3)}</Tag>
                            <Tag>{t("search.semantic")}: {item.semantic_score.toFixed(3)}</Tag>
                            {item.paper.year && <Tag>{item.paper.year}</Tag>}
                            {item.paper.sub_field && <Tag color="geekblue">{item.paper.sub_field}</Tag>}
                          </Space>
                          <Typography.Text type="secondary">{item.snippet}</Typography.Text>
                        </Space>
                      </List.Item>
                    )}
                  />
                </Space>
              </div>
            )
          },
          {
            key: "workflow",
            label: t("tab.workflow"),
            children: (
              <div className="card">
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Space wrap>
                    <Tag color="red">{t("workflow.overdue")}: {taskStats.overdue}</Tag>
                    <Tag color="orange">{t("workflow.review_due")}: {taskStats.reviewDue}</Tag>
                    <Tag color="green">{t("workflow.done")}: {taskStats.done}</Tag>
                    <Button onClick={refreshTasks} loading={taskLoading}>{t("btn.refresh")}</Button>
                  </Space>
                  <List
                    dataSource={tasks}
                    loading={taskLoading}
                    renderItem={(task) => (
                      <List.Item
                        actions={[
                          <Button
                            key="toggle"
                            size="small"
                            onClick={async () => {
                              if (task.status === "done") {
                                await updateTask(task.id, { status: "todo" });
                              } else {
                                await completeReview(task.id);
                              }
                              refreshTasks();
                            }}
                          >
                            {task.status === "done" ? "Undo" : "Done"}
                          </Button>
                        ]}
                      >
                        <Space direction="vertical" size={0}>
                          <Typography.Text>{task.title}</Typography.Text>
                          <Typography.Text type="secondary">
                            #{task.paper_id} {task.due_date ? ` | ${task.due_date}` : ""}
                          </Typography.Text>
                        </Space>
                      </List.Item>
                    )}
                  />
                </Space>
              </div>
            )
          },
          {
            key: "insights",
            label: t("tab.insights"),
            children: (
              <Space direction="vertical" style={{ width: "100%" }} size="middle">
                <div className="card">
                  <Space wrap>
                    <Button
                      onClick={async () => {
                        await generateReport("weekly");
                        await refreshInsights();
                      }}
                    >
                      {t("report.generate_weekly")}
                    </Button>
                    <Button
                      onClick={async () => {
                        await generateReport("monthly");
                        await refreshInsights();
                      }}
                    >
                      {t("report.generate_monthly")}
                    </Button>
                    <Button onClick={refreshInsights} loading={insightLoading}>{t("btn.refresh")}</Button>
                  </Space>
                </div>
                <Card size="small" title={t("insight.trends")}>
                  <div style={{ width: "100%", height: 260 }}>
                    <ResponsiveContainer>
                      <AreaChart data={riverData}>
                        <defs>
                          {topicRiver?.sub_fields.slice(0, 8).map((field, idx) => (
                            <linearGradient id={`river-${idx}`} key={field} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={`hsl(${(idx * 47) % 360}, 70%, 45%)`} stopOpacity={0.7} />
                              <stop offset="95%" stopColor={`hsl(${(idx * 47) % 360}, 70%, 45%)`} stopOpacity={0.12} />
                            </linearGradient>
                          ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="year" />
                        <YAxis />
                        <Tooltip />
                        {topicRiver?.sub_fields.slice(0, 8).map((field, idx) => (
                          <Area
                            key={field}
                            type="monotone"
                            dataKey={field}
                            stackId="1"
                            stroke={`hsl(${(idx * 47) % 360}, 70%, 45%)`}
                            fill={`url(#river-${idx})`}
                          />
                        ))}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <List
                    size="small"
                    dataSource={topicEvolution?.bursts || []}
                    locale={{ emptyText: "-" }}
                    renderItem={(item) => (
                      <List.Item>
                        <Typography.Text>
                          {item.year} · {item.sub_field} · +{item.growth} ({Math.round(item.growth_ratio * 100)}%)
                        </Typography.Text>
                      </List.Item>
                    )}
                  />
                </Card>
                <Card size="small" title={t("insight.sota")}>
                  <Space wrap style={{ marginBottom: 8 }}>
                    <Select
                      allowClear
                      style={{ width: 220 }}
                      value={sotaDataset}
                      placeholder={t("insight.dataset")}
                      options={(sotaBoard?.datasets || []).map((d) => ({ label: d, value: d }))}
                      onChange={(value) => setSotaDataset(value)}
                    />
                    <Button onClick={refreshInsights} loading={insightLoading}>
                      {t("btn.refresh")}
                    </Button>
                  </Space>
                  <Table
                    size="small"
                    rowKey="id"
                    pagination={{ pageSize: 6 }}
                    dataSource={sotaBoard?.items || []}
                    columns={[
                      { title: t("insight.paper"), dataIndex: "title", key: "title", ellipsis: true },
                      { title: t("insight.dataset"), dataIndex: "dataset_name", key: "dataset_name", width: 130 },
                      { title: "P", dataIndex: "precision", key: "precision", width: 76 },
                      { title: "R", dataIndex: "recall", key: "recall", width: 76 },
                      { title: "F1", dataIndex: "f1", key: "f1", width: 76 },
                      { title: t("insight.trigger_f1"), dataIndex: "trigger_f1", key: "trigger_f1", width: 96 },
                      { title: t("insight.argument_f1"), dataIndex: "argument_f1", key: "argument_f1", width: 96 },
                      { title: t("table.year"), dataIndex: "year", key: "year", width: 76 }
                    ]}
                  />
                </Card>
                <Card size="small" title={t("report.list")}>
                  <List
                    dataSource={reports}
                    renderItem={(report) => (
                      <List.Item>
                        <Space direction="vertical" size={0}>
                          <Typography.Text>{report.period_type.toUpperCase()} {report.period_start} - {report.period_end}</Typography.Text>
                          <Typography.Text type="secondary">
                            {report.payload?.next_suggestions?.join("；")}
                          </Typography.Text>
                        </Space>
                      </List.Item>
                    )}
                  />
                </Card>
              </Space>
            )
          },
          {
            key: "quality",
            label: t("tab.quality"),
            children: (
              <Space direction="vertical" style={{ width: "100%" }} size="middle">
                <div className="card">
                  <Space wrap>
                    <Button onClick={refreshQuality} loading={qualityLoading}>{t("btn.refresh")}</Button>
                    <Button
                      onClick={async () => {
                        const res = await autoMerge({ limit: 20, title_threshold: 0.96 });
                        message.success(`merged ${res.merged}`);
                        refreshQuality();
                        loadPapers();
                      }}
                    >
                      {t("quality.auto_merge")}
                    </Button>
                    <InputNumber
                      placeholder={t("quality.source_id")}
                      value={mergeSourceId}
                      onChange={(value) => setMergeSourceId(value ?? undefined)}
                    />
                    <InputNumber
                      placeholder={t("quality.target_id")}
                      value={mergeTargetId}
                      onChange={(value) => setMergeTargetId(value ?? undefined)}
                    />
                    <Button
                      type="primary"
                      onClick={async () => {
                        if (!mergeSourceId || !mergeTargetId) return;
                        await mergePapers({ source_paper_id: mergeSourceId, target_paper_id: mergeTargetId });
                        message.success(t("quality.merge_done"));
                        refreshQuality();
                        loadPapers();
                      }}
                    >
                      {t("quality.merge")}
                    </Button>
                  </Space>
                </div>

                <Card size="small" title={`${t("quality.duplicates")} (${duplicates.length})`}>
                  <List
                    dataSource={duplicates}
                    renderItem={(item) => (
                      <List.Item>
                        <Typography.Text>
                          {item.type} | [{item.paper_ids.join(", ")}] | {Math.round(item.confidence * 100)}%
                        </Typography.Text>
                      </List.Item>
                    )}
                  />
                </Card>

                <Card size="small" title={`${t("quality.conflicts")} (${conflicts.length})`}>
                  <List
                    dataSource={conflicts}
                    renderItem={(item) => (
                      <List.Item>
                        <Typography.Text>
                          DOI: {item.doi} | [{item.paper_ids.join(", ")}] | {item.title_variants.slice(0, 2).join(" / ")}
                        </Typography.Text>
                      </List.Item>
                    )}
                  />
                </Card>

                <Card size="small" title={t("zotero.template")}>
                  <Space direction="vertical" style={{ width: "100%" }}>
                    <Input.TextArea
                      value={zoteroTemplate}
                      onChange={(e) => setZoteroTemplate(e.target.value)}
                      autoSize={{ minRows: 4, maxRows: 10 }}
                    />
                    <Space wrap>
                      <Select
                        value={zoteroDirection}
                        style={{ width: 140 }}
                        onChange={(value) => setZoteroDirection(value)}
                        options={[
                          { label: "both", value: "both" },
                          { label: "pull", value: "pull" },
                          { label: "push", value: "push" }
                        ]}
                      />
                      <Select
                        value={zoteroStrategy}
                        style={{ width: 180 }}
                        onChange={(value) => setZoteroStrategy(value)}
                        options={[
                          { label: "prefer_local", value: "prefer_local" },
                          { label: "prefer_zotero", value: "prefer_zotero" },
                          { label: "manual", value: "manual" }
                        ]}
                      />
                      <InputNumber min={1} max={200} value={zoteroLimit} onChange={(v) => setZoteroLimit(v || 20)} />
                      <Button
                        onClick={async () => {
                          try {
                            const parsed = JSON.parse(zoteroTemplate);
                            await saveZoteroTemplate({ name: "default", mapping: parsed });
                            message.success("template saved");
                          } catch {
                            message.error("invalid JSON");
                          }
                        }}
                      >
                        {t("details.save")}
                      </Button>
                      <Button
                        type="primary"
                        onClick={async () => {
                          const res = await syncZoteroIncremental({
                            direction: zoteroDirection,
                            conflict_strategy: zoteroStrategy,
                            limit: zoteroLimit
                          });
                          message.success(`synced ${res.synced}, conflicts ${res.conflicts}`);
                          refreshZotero();
                        }}
                      >
                        {t("zotero.sync_incremental")}
                      </Button>
                    </Space>
                  </Space>
                </Card>

                <Card size="small" title={t("zotero.logs")}>
                  <List
                    loading={zoteroLoading}
                    dataSource={zoteroLogs}
                    renderItem={(log) => (
                      <List.Item>
                        <Typography.Text>
                          #{log.paper_id} | {log.direction} | {log.action} | {log.status}
                        </Typography.Text>
                      </List.Item>
                    )}
                  />
                </Card>
              </Space>
            )
          }
        ]}
      />
    </>
  );
}

function GraphPage({ t }: { t: (key: string) => string }) {
  return (
    <div className="graph-page">
      <GraphView t={t} standalone />
    </div>
  );
}

function AppShell() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [lang, setLang] = useState<Lang>("zh");
  const t = useMemo(() => createT(lang), [lang]);
  const location = useLocation();

  const loadPapers = async () => {
    setLoading(true);
    try {
      const data = await fetchPapers();
      setPapers(data);
    } catch {
      message.error(t("msg.load_failed"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPapers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpload = async (options: RcCustomRequestOptions) => {
    const file = options.file as File;
    setUploading(true);
    try {
      await uploadPaper(file);
      message.success(t("msg.uploaded"));
      await loadPapers();
      options.onSuccess?.({}, file as any);
    } catch (err: any) {
      message.error(err?.response?.data?.detail || t("msg.upload_failed"));
      options.onError?.(err);
    } finally {
      setUploading(false);
    }
  };

  const stats = useMemo(() => {
    const total = papers.length;
    const read = papers.filter((p) => p.read_status === 1).length;
    const unread = total - read;
    return { total, read, unread };
  }, [papers]);

  return (
    <ConfigProvider locale={lang === "zh" ? zhCN : enUS}>
      <Layout className="app-shell">
        <Header className="app-header">
          <div className="app-header-inner">
            <div className="app-brand">
              <span className="app-kicker">{t("app.kicker")}</span>
              <Typography.Title level={2} className="app-title">
                {t("app.title")}
              </Typography.Title>
              <Typography.Text className="app-subtitle">{t("app.subtitle")}</Typography.Text>
              <div className="app-pills">
                <NavLink className={({ isActive }) => `app-pill nav-pill ${isActive ? "is-active" : ""}`} to="/">
                  {t("nav.home")}
                </NavLink>
                <NavLink className={({ isActive }) => `app-pill nav-pill ${isActive ? "is-active" : ""}`} to="/graph">
                  {t("nav.graph")}
                </NavLink>
              </div>
            </div>
            <div className="app-actions">
              <div className="lang-toggle">
                <button className={`lang-btn ${lang === "zh" ? "is-active" : ""}`} onClick={() => setLang("zh")}>
                  中文
                </button>
                <button className={`lang-btn ${lang === "en" ? "is-active" : ""}`} onClick={() => setLang("en")}>
                  EN
                </button>
              </div>
              {location.pathname !== "/graph" && (
                <div className="app-stats">
                  <div className="stat">
                    <span className="stat-label">{t("stats.total")}</span>
                    <span className="stat-value">{stats.total}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">{t("stats.read")}</span>
                    <span className="stat-value">{stats.read}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">{t("stats.unread")}</span>
                    <span className="stat-value">{stats.unread}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </Header>
        <Content className={`app-content ${location.pathname === "/graph" ? "is-graph-page" : ""}`}>
          <Routes>
            <Route
              path="/"
              element={
                <DashboardPage
                  t={t}
                  lang={lang}
                  papers={papers}
                  loading={loading}
                  uploading={uploading}
                  loadPapers={loadPapers}
                  onUpload={handleUpload}
                />
              }
            />
            <Route path="/graph" element={<GraphPage t={t} />} />
          </Routes>
        </Content>
      </Layout>
    </ConfigProvider>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
