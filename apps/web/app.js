"use strict";

// ---------------------------------------------------------------- tabs ---

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "facts") loadFacts();
    if (btn.dataset.tab === "relations") loadRelations();
    if (btn.dataset.tab === "review") loadReview();
  });
});

// -------------------------------------------------------------- upload ---

const pdfInput = document.getElementById("pdf-input");
const uploadBtn = document.getElementById("upload-btn");
const progressLog = document.getElementById("upload-progress");
const uploadSummary = document.getElementById("upload-summary");

function logLine(text, cls) {
  const div = document.createElement("div");
  div.className = "line" + (cls ? ` ${cls}` : "");
  div.textContent = text;
  progressLog.appendChild(div);
  progressLog.scrollTop = progressLog.scrollHeight;
}

const STAGE_LABELS = {
  started: (d) => `Starting ingest of ${d.pdf_path}…`,
  blocks_extracted: (d) => `Extracted ${d.count} block(s) from the PDF.`,
  deriving_doc_context: () => `Deriving document context (1 LLM call)…`,
  doc_context: (d) => `Document context: ${d.doc_context.publisher || "?"} — ${d.doc_context.document_type || "?"} (${d.doc_context.reporting_period || "?"})`,
  extracting: () => `Extracting facts (prose + table)…`,
  extraction_complete: (d) => `Extraction complete: ${d.facts} fact(s), ${d.quarantined} quarantined, ${d.failed_batches} batch failure(s).`,
  facts_persisted: (d) => `Persisted ${d.count} fact(s).`,
  candidates_found: (d) => `Found ${d.count} new candidate pair(s) to reconcile.`,
  relation: (d) => `Reconciled ${d.done}/${d.total}: ${d.type} / ${d.reason_code}`,
  already_ingested: (d) => `Document ${d.doc_id} was already ingested — skipping re-extraction.`,
  error: (d) => `Error: ${d.message}`,
  done: () => `Done.`,
};

async function uploadAndIngest() {
  const file = pdfInput.files[0];
  if (!file) {
    alert("Choose a PDF file first.");
    return;
  }
  uploadBtn.disabled = true;
  progressLog.hidden = false;
  progressLog.innerHTML = "";
  uploadSummary.hidden = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/documents", { method: "POST", body: formData });
    if (!response.ok || !response.body) {
      logLine(`Upload failed: HTTP ${response.status}`, "error");
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop(); // last chunk may be incomplete
      for (const raw of events) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        const label = STAGE_LABELS[payload.stage];
        const text = label ? label(payload) : `${payload.stage}: ${JSON.stringify(payload)}`;
        logLine(text, payload.stage === "error" ? "error" : payload.stage === "done" ? "done" : "");
        if (payload.stage === "done" && payload.result) {
          showUploadSummary(payload.result);
        }
      }
    }
  } catch (err) {
    logLine(`Upload failed: ${err}`, "error");
  } finally {
    uploadBtn.disabled = false;
  }
}

function showUploadSummary(result) {
  const relCounts = Object.entries(result.relations_by_type)
    .filter(([, n]) => n > 0)
    .map(([type, n]) => `${type}: ${n}`)
    .join(", ") || "none";
  uploadSummary.hidden = false;
  uploadSummary.innerHTML = `
    <strong>${result.doc_id}</strong><br>
    Facts: ${result.facts} &middot; Quarantined: ${result.quarantined}<br>
    Relations — ${relCounts}
  `;
}

uploadBtn.addEventListener("click", uploadAndIngest);

// --------------------------------------------------------------- facts ---

const factsTableBody = document.querySelector("#facts-table tbody");
const factDetail = document.getElementById("fact-detail");
let currentFacts = [];

async function populateDocFilter() {
  const select = document.getElementById("facts-doc-filter");
  const docs = await fetchJSON("/documents");
  const existing = new Set([...select.options].map((o) => o.value));
  for (const doc of docs) {
    if (existing.has(doc.doc_id)) continue;
    const opt = document.createElement("option");
    opt.value = doc.doc_id;
    opt.textContent = doc.doc_id;
    select.appendChild(opt);
  }
}

async function loadFacts() {
  await populateDocFilter();
  const params = new URLSearchParams();
  const docId = document.getElementById("facts-doc-filter").value;
  const subject = document.getElementById("facts-subject-filter").value.trim();
  const measure = document.getElementById("facts-measure-filter").value.trim();
  if (docId) params.set("doc_id", docId);
  if (subject) params.set("subject", subject);
  if (measure) params.set("measure", measure);

  currentFacts = await fetchJSON(`/facts?${params}`);
  factsTableBody.innerHTML = "";
  for (const fact of currentFacts) {
    const tr = document.createElement("tr");
    const period = fact.qualifiers.period_label || fact.qualifiers.period_end || "";
    tr.innerHTML = `
      <td>${escapeHtml(fact.subject.surface_form)}</td>
      <td>${escapeHtml(fact.measure.surface_form)}</td>
      <td>${escapeHtml(fact.value.raw)}</td>
      <td>${escapeHtml(period)}</td>
      <td>${fact.evidence.page_no}</td>
    `;
    tr.addEventListener("click", () => selectFact(fact, tr));
    factsTableBody.appendChild(tr);
  }
  factDetail.hidden = true;
}

async function selectFact(fact, row) {
  document.querySelectorAll("#facts-table tbody tr").forEach((r) => r.classList.remove("selected"));
  row.classList.add("selected");
  factDetail.hidden = false;

  const fields = document.getElementById("fact-detail-fields");
  fields.innerHTML = `
    <dt>Qualifiers</dt><dd>${escapeHtml(JSON.stringify(fact.qualifiers))}</dd>
    <dt>Confidence</dt><dd>${fact.confidence.extraction}</dd>
    <dt>Provenance</dt><dd>${escapeHtml(fact.provenance.model)}</dd>
    <dt>Quote</dt><dd>${escapeHtml(fact.evidence.quote)}</dd>
  `;

  await renderPageWithHighlight(fact.doc_id, fact.evidence.page_no, fact.evidence.bbox, "fact-page-img", "fact-highlight-overlay");
}

document.getElementById("facts-refresh").addEventListener("click", loadFacts);

// ----------------------------------------------------------- relations ---

const relationsTableBody = document.querySelector("#relations-table tbody");
const relationDetail = document.getElementById("relation-detail");

async function loadRelations() {
  const params = new URLSearchParams();
  const type = document.getElementById("relations-type-filter").value;
  const reason = document.getElementById("relations-reason-filter").value.trim();
  if (type) params.set("type", type);
  if (reason) params.set("reason_code", reason);

  const relations = await fetchJSON(`/relations?${params}`);
  relationsTableBody.innerHTML = "";
  for (const rel of relations) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="badge badge-${rel.type}">${rel.type}</span></td>
      <td>${rel.reason_code}</td>
      <td class="muted">${rel.fact_a_id.slice(0, 8)}…</td>
      <td class="muted">${rel.fact_b_id.slice(0, 8)}…</td>
      <td>${rel.confidence.toFixed(2)}</td>
    `;
    tr.addEventListener("click", () => selectRelation(rel.id, tr));
    relationsTableBody.appendChild(tr);
  }
  relationDetail.hidden = true;
}

async function selectRelation(relationId, row) {
  document.querySelectorAll("#relations-table tbody tr").forEach((r) => r.classList.remove("selected"));
  row.classList.add("selected");

  const detail = await fetchJSON(`/relations/${relationId}`);
  relationDetail.hidden = false;

  const badge = document.getElementById("relation-verdict-badge");
  badge.textContent = detail.relation.type;
  badge.className = `badge badge-${detail.relation.type}`;
  document.getElementById("relation-reason-code").textContent = detail.relation.reason_code;
  document.getElementById("relation-explanation").textContent = detail.relation.explanation;

  fillFactQuoteBox("a", detail.fact_a);
  fillFactQuoteBox("b", detail.fact_b);
}

function fillFactQuoteBox(side, fact) {
  document.getElementById(`relation-fact-${side}-label`).textContent =
    `${fact.subject.surface_form} — ${fact.measure.surface_form}`;
  document.getElementById(`relation-fact-${side}-quote`).textContent = fact.evidence.quote;
  document.getElementById(`relation-fact-${side}-fields`).innerHTML = `
    <dt>Value</dt><dd>${escapeHtml(fact.value.raw)}</dd>
    <dt>Qualifiers</dt><dd>${escapeHtml(JSON.stringify(fact.qualifiers))}</dd>
    <dt>Page</dt><dd>${fact.evidence.page_no}</dd>
  `;
}

document.getElementById("relations-refresh").addEventListener("click", loadRelations);

// -------------------------------------------------------------- review ---

async function loadReview() {
  const entries = await fetchJSON("/review");
  const tbody = document.querySelector("#review-table tbody");
  tbody.innerHTML = "";
  for (const entry of entries) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(entry.doc_id)}</td>
      <td>${escapeHtml(entry.fact.subject.surface_form)}</td>
      <td>${escapeHtml(entry.fact.measure.surface_form)}</td>
      <td>${escapeHtml(entry.fact.evidence.quote)}</td>
      <td>${escapeHtml(entry.reason)}</td>
      <td>${entry.score.toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  }
}

// --------------------------------------------------- page render + bbox --

async function renderPageWithHighlight(docId, pageNo, bboxList, imgId, overlayId) {
  const img = document.getElementById(imgId);
  const overlay = document.getElementById(overlayId);
  overlay.innerHTML = "";

  const response = await fetch(`/documents/${encodeURIComponent(docId)}/page/${pageNo}.png`);
  if (!response.ok) {
    img.removeAttribute("src");
    return;
  }
  const widthPt = parseFloat(response.headers.get("X-Page-Width-Pt"));
  const heightPt = parseFloat(response.headers.get("X-Page-Height-Pt"));
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);

  img.onload = () => {
    // Scale PDF-point bbox coordinates onto however large the <img> is
    // actually displayed (not the PNG's raw pixel size) — see
    // apps/api/routes/documents.py for why the headers carry point-space
    // dimensions rather than baking a fixed DPI into this file.
    const scaleX = img.clientWidth / widthPt;
    const scaleY = img.clientHeight / heightPt;
    overlay.style.width = `${img.clientWidth}px`;
    overlay.style.height = `${img.clientHeight}px`;
    overlay.innerHTML = "";
    for (const [x0, y0, x1, y1] of bboxList || []) {
      const box = document.createElement("div");
      box.className = "highlight-box";
      box.style.left = `${x0 * scaleX}px`;
      box.style.top = `${y0 * scaleY}px`;
      box.style.width = `${(x1 - x0) * scaleX}px`;
      box.style.height = `${(y1 - y0) * scaleY}px`;
      overlay.appendChild(box);
    }
  };
  img.src = url;
}

// ---------------------------------------------------------------- util ---

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}
