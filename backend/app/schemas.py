from __future__ import annotations

from pydantic import BaseModel


class Paper(BaseModel):
    id: int
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    journal_conf: str | None = None
    ccf_level: str | None = None
    source_type: str | None = None
    sub_field: str | None = None
    read_status: int = 0
    file_path: str | None = None
    doi: str | None = None
    abstract: str | None = None
    s2_paper_id: str | None = None
    s2_corpus_id: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None
    citation_velocity: float | None = None
    url: str | None = None
    venue: str | None = None
    keywords: str | None = None
    cluster_id: str | None = None
    zotero_item_key: str | None = None
    zotero_library: str | None = None
    zotero_item_id: int | None = None
    file_hash: str | None = None
    summary_one: str | None = None
    refs_parsed_at: int | None = None
    created_at: int | None = None
    updated_at: int | None = None
    proposed_method_name: str | None = None
    dynamic_tags: str | None = None
    embedding: str | None = None


class PaperCreate(BaseModel):
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    journal_conf: str | None = None
    ccf_level: str | None = None
    source_type: str | None = None
    sub_field: str | None = None
    read_status: int = 0
    file_path: str | None = None
    doi: str | None = None
    abstract: str | None = None
    s2_paper_id: str | None = None
    s2_corpus_id: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None
    citation_velocity: float | None = None
    url: str | None = None
    venue: str | None = None
    keywords: str | None = None
    cluster_id: str | None = None
    zotero_item_key: str | None = None
    zotero_library: str | None = None
    zotero_item_id: int | None = None
    file_hash: str | None = None
    summary_one: str | None = None
    refs_parsed_at: int | None = None
    created_at: int | None = None
    updated_at: int | None = None
    proposed_method_name: str | None = None
    dynamic_tags: str | None = None
    embedding: str | None = None


class PaperUpdate(BaseModel):
    read_status: int | None = None
    zotero_item_key: str | None = None
    zotero_library: str | None = None
    zotero_item_id: int | None = None
    sub_field: str | None = None
    summary_one: str | None = None
    proposed_method_name: str | None = None
    dynamic_tags: str | None = None


class PaperNote(BaseModel):
    paper_id: int
    method: str | None = None
    datasets: str | None = None
    conclusions: str | None = None
    reproducibility: str | None = None
    risks: str | None = None
    notes: str | None = None
    created_at: int | None = None
    updated_at: int | None = None


class PaperNoteUpdate(BaseModel):
    method: str | None = None
    datasets: str | None = None
    conclusions: str | None = None
    reproducibility: str | None = None
    risks: str | None = None
    notes: str | None = None


class ReadingTask(BaseModel):
    id: int
    paper_id: int
    title: str
    status: str = "todo"
    priority: int = 2
    due_date: str | None = None
    next_review_at: int | None = None
    interval_days: int = 1
    last_review_at: int | None = None
    created_at: int | None = None
    updated_at: int | None = None


class ReadingTaskCreate(BaseModel):
    paper_id: int
    title: str
    priority: int = 2
    due_date: str | None = None
    next_review_at: int | None = None
    interval_days: int = 1


class ReadingTaskUpdate(BaseModel):
    title: str | None = None
    status: str | None = None
    priority: int | None = None
    due_date: str | None = None
    next_review_at: int | None = None
    interval_days: int | None = None


class Experiment(BaseModel):
    id: int
    paper_id: int
    name: str | None = None
    model: str | None = None
    params_json: str | None = None
    metrics_json: str | None = None
    result_summary: str | None = None
    artifact_path: str | None = None
    dataset_name: str | None = None
    trigger_f1: float | None = None
    argument_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    created_at: int | None = None
    updated_at: int | None = None


class ExperimentCreate(BaseModel):
    paper_id: int
    name: str | None = None
    model: str | None = None
    params_json: str | None = None
    metrics_json: str | None = None
    result_summary: str | None = None
    artifact_path: str | None = None
    dataset_name: str | None = None
    trigger_f1: float | None = None
    argument_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class ExperimentUpdate(BaseModel):
    name: str | None = None
    model: str | None = None
    params_json: str | None = None
    metrics_json: str | None = None
    result_summary: str | None = None
    artifact_path: str | None = None
    dataset_name: str | None = None
    trigger_f1: float | None = None
    argument_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class Subfield(BaseModel):
    id: int
    name: str
    description: str | None = None
    active: int = 1


class SubfieldCreate(BaseModel):
    name: str
    description: str | None = None
    active: int = 1


class SubfieldUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    active: int | None = None
