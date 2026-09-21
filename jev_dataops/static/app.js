/* JEV DataOps — dependency-free, same-origin workspace. */
"use strict";

const $ = (id) => document.getElementById(id);
const emptyRunList = $("run-list").firstElementChild.cloneNode(true);
const state = {
  datasets: [], runs: [], datasetId: null, runId: null, health: null,
  token: sessionStorage.getItem("jev_api_token") || "", loading: false,
  uploading: false, submitting: false, polling: false, timer: null,
  toastTimer: null, logsSignature: "", artifactsSignature: "", actionPending: false, settlePending: false,
};
const statuses = { queued: "等待中", running: "运行中", completed: "已完成", failed: "失败", cancelled: "已取消" };
const stages = [
  { key: "upload", label: "上传数据" }, { key: "screening", label: "智能筛选" },
  { key: "data_evaluation", label: "数据评估" }, { key: "training", label: "自动训练" },
  { key: "model_evaluation", label: "模型评估" },
];
const number = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString("zh-CN") : "—";
const formatMetric = (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
const isActive = (run) => run && ["running", "queued"].includes(run.status);
function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}
function displayDate(value, timeOnly = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", timeOnly ? { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false } : { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(date);
}
function fileSize(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let size = Number(value), unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit++; }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
function showToast(message, error = false) {
  clearTimeout(state.toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("toast-error", error);
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 6500 : 3500);
}
function errorMessage(error) {
  return error instanceof TypeError ? "无法连接到服务，请检查服务是否启动，然后重试。" : error.message || "操作失败，请重试。";
}
function showError(error) {
  const message = errorMessage(error);
  $("global-error").textContent = message;
  $("global-error").hidden = false;
  showToast(message, true);
}
function setConnection(online) {
  $("connection").classList.toggle("offline", !online);
  $("connection-label").textContent = online ? "服务已连接" : "连接中断";
}
async function request(path, { method = "GET", body, raw = false } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body && !(body instanceof FormData)) { headers["Content-Type"] = "application/json"; body = JSON.stringify(body); }
  const response = await fetch(path, { method, body, headers, credentials: "same-origin" });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      const detail = payload.detail || payload.message;
      if (typeof detail === "string") message = detail;
      else if (Array.isArray(detail)) message = detail.map((item) => `${(item.loc || []).filter((key) => key !== "body").join(".")}: ${item.msg}`).join("；");
    } catch (_) { /* Keep the HTTP error when the response is not JSON. */ }
    if (response.status === 401 || response.status === 403) message = "访问令牌无效或缺失。请在「连接设置」中输入服务端配置的令牌。";
    throw new Error(message);
  }
  return raw ? response : response.json();
}
function updateStartState() {
  $("start-button").disabled = !state.datasetId || state.loading || state.uploading || state.submitting;
  $("start-button").querySelector("span").textContent = state.submitting ? "正在创建工作流…" : "启动工作流";
  $("start-hint").textContent = state.datasetId ? "任务与产物保存在服务端，可随时返回查看" : "先添加一个数据集，即可开始";
}
function applyHealth(health) {
  state.health = health;
  $("version").textContent = `v${health.version || "0.1"}`;
  for (const [value, name] of [["openrouter", "OpenRouter"], ["typesafe", "TypeSafe"]]) {
    const option = $("provider").querySelector(`option[value="${value}"]`);
    option.disabled = !health.providers?.[value];
    option.textContent = `JEV · ${name}${option.disabled ? "（未配置）" : ""}`;
  }
  const trainer = $("trainer").querySelector('option[value="huggingface"]');
  trainer.disabled = !health.training?.huggingface;
  trainer.textContent = `Hugging Face · LoRA${trainer.disabled ? "（未安装）" : ""}`;
  if ($( "provider").selectedOptions[0]?.disabled) $("provider").value = "demo";
  if ($( "trainer").selectedOptions[0]?.disabled) $("trainer").value = "demo";
  if (health.max_upload_mb) $("upload-subtitle").textContent = `JSONL / CSV · 最大 ${number(health.max_upload_mb)} MB · 流式读取`;
  updateModeNotice();
}
function updateModeNotice() {
  const provider = $("provider").value;
  const trainer = $("trainer").value;
  const autoTrain = $("auto-train").checked;
  const title = $("mode-notice").querySelector("strong");
  const description = $("mode-notice").querySelector("p > span");
  $("mode-notice").classList.toggle("real-mode", provider !== "demo");
  $("trainer").disabled = !autoTrain;
  if (provider === "demo") {
    title.textContent = autoTrain && trainer === "huggingface" ? "本地规则筛选 + 大模型训练" : "当前为 Demo 模式";
    description.textContent = autoTrain && trainer === "huggingface"
      ? `筛选使用本地规则，不调用 JEV；保留数据将用于 ${state.health?.training?.base_model || "已配置模型"} 的 LoRA 训练。`
      : autoTrain ? "本地规则筛选与字节二元统计模型验证，不调用 JEV，也不训练大模型。" : "仅进行本地规则筛选与数据评估，不调用 JEV，也不训练模型。";
  } else {
    title.textContent = "已启用 JEV 在线筛选";
    description.textContent = !autoTrain ? "数据内容将发送至所选第三方 JEV 服务；请仅提交有权共享的数据。请求上限用于约束 API 预算。"
      : trainer === "demo" ? "数据内容将发送至所选第三方 JEV 服务；请仅提交有权共享的数据。后续验证字节统计模型，不训练大模型。"
      : `数据内容发送至第三方 JEV 服务，请仅提交有权共享的数据。随后训练 ${state.health?.training?.base_model || "已配置模型"} 的 LoRA 适配器。`;
  }
}
function renderDatasets() {
  const select = $("dataset-select");
  select.replaceChildren();
  if (!state.datasets.length) select.append(el("option", "", "还没有数据集，请先上传"));
  for (const dataset of state.datasets) {
    const option = el("option", "", `${dataset.name} · ${number(dataset.rows)} 行`);
    option.value = dataset.id;
    select.append(option);
  }
  select.disabled = !state.datasets.length;
  if (!state.datasets.some((dataset) => dataset.id === state.datasetId)) state.datasetId = state.datasets[0]?.id || null;
  select.value = state.datasetId || "";
  $("nav-dataset-count").textContent = number(state.datasets.length);
  renderDatasetPreview();
  updateStartState();
}
function renderDatasetPreview() {
  const dataset = state.datasets.find((item) => item.id === state.datasetId);
  $("dataset-summary").hidden = !dataset;
  $("empty-preview").hidden = Boolean(dataset);
  $("preview-wrap").hidden = !dataset || !dataset.preview?.length;
  if (!dataset) return;
  $("dataset-name").textContent = dataset.name;
  $("dataset-name").title = dataset.name;
  $("dataset-meta").textContent = `${number(dataset.rows)} 行 · ${fileSize(dataset.size)}`;
  const table = $("preview-table");
  const head = table.querySelector("thead"), body = table.querySelector("tbody");
  head.replaceChildren(); body.replaceChildren();
  const rows = (dataset.preview || []).slice(0, 5);
  const keys = [...new Set(rows.flatMap((row) => Object.keys(row || {})))].slice(0, 5);
  const header = el("tr");
  const rowHeader = el("th", "", "#"); rowHeader.scope = "col"; header.append(rowHeader);
  for (const key of keys) { const cell = el("th", "", key); cell.scope = "col"; header.append(cell); }
  head.append(header);
  rows.forEach((row, index) => {
    const tr = el("tr"); tr.append(el("td", "", String(index + 1).padStart(2, "0")));
    for (const key of keys) {
      const value = row[key] === undefined ? "—" : typeof row[key] === "object" ? JSON.stringify(row[key]) : String(row[key]);
      const td = el("td", "", value); td.title = value; tr.append(td);
    }
    body.append(tr);
  });
}
function renderRunList() {
  const list = $("run-list");
  const focusedRun = list.contains(document.activeElement) ? document.activeElement.dataset.runId : null;
  $("history-count").textContent = number(state.runs.length);
  $("nav-run-count").textContent = number(state.runs.length);
  $("run-total").textContent = `${number(state.runs.length)} RUN${state.runs.length === 1 ? "" : "S"}`;
  if (!state.runs.length) { list.replaceChildren(emptyRunList.cloneNode(true)); return; }
  const fragment = document.createDocumentFragment();
  for (const run of state.runs) {
    const button = el("button", `run-list-button${run.id === state.runId ? " selected" : ""}`);
    button.type = "button"; button.dataset.runId = run.id;
    button.setAttribute("aria-pressed", String(run.id === state.runId));
    button.append(el("div", "run-list-name", run.name || `Run ${run.id.slice(0, 8)}`));
    const meta = el("div", "run-list-meta");
    meta.append(el("span", `status-badge status-${Object.hasOwn(statuses, run.status) ? run.status : "queued"}`, statuses[run.status] || run.status), el("span", "", displayDate(run.created_at)));
    button.append(meta, el("div", "run-list-provider", `${run.config?.provider === "demo" ? "DEMO · 本地验证" : `JEV · ${run.config?.provider || "—"}`} / ${run.id.slice(0, 8)}`));
    button.addEventListener("click", () => selectRun(run.id));
    fragment.append(button);
  }
  list.replaceChildren(fragment);
  if (focusedRun) [...list.querySelectorAll("button")].find((button) => button.dataset.runId === focusedRun)?.focus({ preventScroll: true });
}
function renderStages(run) {
  const track = $("stage-track"); track.replaceChildren();
  const complete = run.status === "completed";
  const noTraining = run.config?.auto_train === false;
  const index = complete ? (noTraining ? 2 : 4) : Math.max(0, stages.findIndex((stage) => stage.key === run.stage));
  stages.forEach((stage, position) => {
    const skipped = noTraining && position > 2;
    const done = !skipped && (position < index || complete && position <= index);
    const active = !skipped && !complete && position === index;
    const failed = active && ["failed", "cancelled"].includes(run.status);
    const item = el("div", `run-stage${done ? " done" : ""}${active ? " active" : ""}${failed ? " failed" : ""}`);
    item.append(el("span", "run-stage-symbol", skipped ? "–" : done ? "✓" : failed ? "!" : String(position + 1).padStart(2, "0")), el("span", "", skipped ? `${stage.label} · 跳过` : stage.label));
    if (active) item.setAttribute("aria-current", "step");
    track.append(item);
  });
}
function renderProgress(run) {
  const progress = run.progress || {};
  const dataset = state.datasets.find((item) => item.id === run.dataset_id);
  let processed = Number(progress.processed ?? run.data_report?.processed ?? 0);
  let total = Number(progress.total ?? run.data_report?.counts?.total ?? dataset?.rows ?? 0);
  let label = stages.find((stage) => stage.key === run.stage)?.label || "工作流";
  if (["training", "model_evaluation"].includes(run.stage) && progress.max_steps) {
    processed = Number(progress.step || 0); total = Number(progress.max_steps); label = "训练步数";
  }
  if (progress.stage === "loading_model") label = "正在加载基础模型";
  if (progress.stage === "splitting") label = "正在构建独立数据集";
  if (["evaluating", "evaluating_baseline"].includes(progress.stage)) label = progress.stage === "evaluating_baseline" ? "评估训练前的模型" : "评估训练后的模型";
  if (run.status === "queued") label = "任务已创建，等待本地执行";
  if (run.status === "completed") { label = "工作流已完成"; processed = total; }
  if (run.status === "cancelled") label += " · 已取消";
  if (run.status === "failed") label += " · 运行失败";
  $("progress-label").textContent = label;
  $("progress-numbers").textContent = total ? `${number(processed)} / ${number(total)}` : isActive(run) ? "处理中…" : "—";
  if (!total && isActive(run)) $("run-progress").removeAttribute("value");
  else $("run-progress").value = run.status === "completed" ? 100 : total > 0 ? Math.min(100, Math.max(0, processed / total * 100)) : 0;
  $("run-progress").setAttribute("aria-label", label);
}
function renderModelReport(report) {
  $("model-report-section").hidden = !report;
  if (!report) return;
  const demo = report.is_llm === false || report.trainer === "demo" || report.mode === "demo";
  $("evaluation-kind").textContent = demo ? "DEMO · BYTE BIGRAM" : "LLM · HELD-OUT TEST";
  const metrics = $("model-metrics"); metrics.replaceChildren();
  for (const [label, value, perplexity] of [["训练前 · Test NLL ↓", report.baseline_loss, report.baseline_perplexity], ["训练后 · Test NLL ↓", report.trained_loss, report.trained_perplexity]]) {
    const metric = el("div", "model-metric"); metric.append(el("span", "", label), el("strong", "", formatMetric(value)));
    if (typeof perplexity === "number") metric.append(el("small", "", `Perplexity ${formatMetric(perplexity)}`));
    metrics.append(metric);
  }
  const notes = [];
  if (typeof report.delta_loss === "number" && Number.isFinite(report.delta_loss)) notes.push(`Δ loss ${report.delta_loss > 0 ? "+" : ""}${report.delta_loss.toFixed(4)}（负数表示下降）`);
  if (report.split_counts) notes.push(`训练 / 验证 / 测试：${["train", "validation", "test"].map((key) => number(report.split_counts[key] || 0)).join(" / ")}`);
  notes.push(demo ? "Demo 为 UTF-8 字节二元统计模型，不是大模型；指标不能与 LLM token loss 直接比较。" : "在相同独立测试集上对比模型损失；该指标不代表任务准确率或生产效果。");
  $("evaluation-note").textContent = notes.join(" · ");
}
function renderLogs(run) {
  const logs = run.logs || [];
  const signature = `${run.id}:${JSON.stringify(logs)}`;
  if (signature === state.logsSignature) return;
  state.logsSignature = signature;
  const container = $("run-logs");
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 35;
  const previousTop = container.scrollTop;
  const fragment = document.createDocumentFragment();
  if (!logs.length) fragment.append(el("span", "", "等待任务输出…"));
  for (const log of logs) {
    const row = el("div", "log-row"); row.append(el("span", "log-time", displayDate(log.time, true)), el("span", "log-text", log.message ?? "")); fragment.append(row);
  }
  container.replaceChildren(fragment);
  container.scrollTop = nearBottom ? container.scrollHeight : previousTop;
  $("log-count").textContent = `${logs.length} entries`;
}
function renderArtifacts(run) {
  const artifacts = run.artifacts || [];
  $("artifact-section").hidden = !artifacts.length;
  const signature = `${run.id}:${JSON.stringify(artifacts)}`;
  if (signature === state.artifactsSignature) return;
  state.artifactsSignature = signature;
  $("artifacts").replaceChildren();
  for (const artifact of artifacts) {
    const button = el("button", "artifact-button"); button.type = "button";
    button.append(el("span", "", "↓"), el("span", "", artifact.name));
    button.title = `${artifact.name}${artifact.size !== undefined ? ` · ${fileSize(artifact.size)}` : ""}`;
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const url = new URL(artifact.url, location.origin);
        if (url.origin !== location.origin || !url.pathname.startsWith("/api/runs/")) throw new Error("无效的产物下载地址。");
        if (!state.token) {
          // Let the browser stream large local downloads directly to disk.
          const link = el("a"); link.href = url.pathname + url.search; link.download = artifact.name.split("/").pop() || "artifact";
          document.body.append(link); link.click(); link.remove(); return;
        }
        const response = await request(url.pathname + url.search, { raw: true });
        const blob = await response.blob();
        const objectURL = URL.createObjectURL(blob);
        const link = el("a"); link.href = objectURL; link.download = artifact.name.split("/").pop() || "artifact";
        document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(objectURL), 10000);
      } catch (error) { showError(error); }
      finally { button.disabled = false; }
    });
    $("artifacts").append(button);
  }
}
function renderRun() {
  const run = state.runs.find((item) => item.id === state.runId);
  $("run-empty").hidden = Boolean(run);
  $("run-content").hidden = !run;
  if (!run) return;
  $("run-name").textContent = run.name || `Run ${run.id.slice(0, 8)}`;
  $("run-status").textContent = statuses[run.status] || run.status;
  $("run-status").className = `status-badge status-${Object.hasOwn(statuses, run.status) ? run.status : "queued"}`;
  const mode = run.config?.provider === "demo" ? "Demo 规则筛选" : `JEV ${run.config?.provider || ""}`;
  $("run-meta").textContent = `${run.id.slice(0, 8)} · ${displayDate(run.created_at)} · ${mode}`;
  $("cancel-button").hidden = !isActive(run); $("cancel-button").disabled = state.actionPending;
  $("retry-button").hidden = !["failed", "cancelled"].includes(run.status); $("retry-button").disabled = state.actionPending;
  $("run-error").hidden = !run.error; $("run-error").textContent = run.error || "";
  renderStages(run); renderProgress(run);
  for (const key of ["keep", "review", "reject"]) $( `count-${key}`).textContent = run.counts?.[key] !== undefined ? number(run.counts[key]) : "—";
  renderModelReport(run.model_report);
  $("data-report-section").hidden = !run.data_report;
  $("data-report").textContent = run.data_report ? JSON.stringify(run.data_report, null, 2) : "";
  renderLogs(run); renderArtifacts(run);
}
function upsertRun(run) {
  const position = state.runs.findIndex((item) => item.id === run.id);
  if (position < 0) state.runs.unshift(run); else state.runs[position] = run;
}
async function selectRun(id) {
  state.runId = id;
  renderRunList(); renderRun();
  try {
    const run = await request(`/api/runs/${encodeURIComponent(id)}`);
    upsertRun(run);
    if (state.runId === id) { renderRunList(); renderRun(); }
    schedulePoll();
  } catch (error) { showError(error); }
}
async function loadWorkspace() {
  if (state.loading) return;
  state.loading = true; $("refresh-button").disabled = true; updateStartState();
  try {
    const [health, datasets, runs] = await Promise.all([request("/api/health"), request("/api/datasets"), request("/api/runs")]);
    applyHealth(health);
    state.datasets = datasets;
    state.runs = runs;
    if (!state.runs.some((run) => run.id === state.runId)) state.runId = state.runs[0]?.id || null;
    renderDatasets(); renderRunList(); renderRun();
    $("global-error").hidden = true; setConnection(true);
  } catch (error) { setConnection(false); showError(error); }
  finally { state.loading = false; $("refresh-button").disabled = false; updateStartState(); schedulePoll(); }
}
function schedulePoll(delay = 1500) {
  clearTimeout(state.timer);
  if (state.runs.some(isActive) || state.settlePending) state.timer = setTimeout(pollRuns, delay);
}
async function pollRuns() {
  if (state.polling) return schedulePoll();
  if (document.hidden) return schedulePoll(5000);
  state.polling = true;
  let failed = false;
  try {
    const runs = await request("/api/runs");
    const priorSelected = state.runs.find((run) => run.id === state.runId);
    // One follow-up fetch also picks up artifacts finalized after terminal status.
    state.settlePending = state.runs.some((previous) => isActive(previous) && runs.some((current) => current.id === previous.id && !isActive(current)));
    state.runs = runs;
    renderRunList(); renderRun(); setConnection(true);
    const current = state.runs.find((run) => run.id === state.runId);
    if (isActive(priorSelected) && current && !isActive(current)) showToast(`工作流${statuses[current.status] || current.status}`, current.status === "failed");
  } catch (_) { failed = true; setConnection(false); }
  finally { state.polling = false; schedulePoll(failed ? 5000 : 1500); }
}
function setUploading(uploading) {
  state.uploading = uploading;
  $("drop-zone").classList.toggle("uploading", uploading);
  $("drop-zone").setAttribute("aria-busy", String(uploading));
  $("file-input").disabled = uploading; $("example-button").disabled = uploading;
  $("upload-progress").hidden = !uploading;
  $("upload-title").textContent = uploading ? "正在上传并检查数据…" : "将数据文件拖拽到这里";
  updateStartState();
}
async function uploadDataset(file, example = false) {
  if (state.uploading) return;
  if (!example) {
    if (!file) return;
    if (!/\.(jsonl|csv)$/i.test(file.name)) return showToast("请选择 UTF-8 编码的 .jsonl 或 .csv 文件。", true);
    if (state.health?.max_upload_mb && file.size > state.health.max_upload_mb * 1024 * 1024) return showToast(`文件超过 ${number(state.health.max_upload_mb)} MB 的服务端限制。`, true);
    if (!file.size) return showToast("文件为空，请选择包含记录的数据集。", true);
  }
  setUploading(true);
  try {
    const form = new FormData(); if (file) form.append("file", file);
    const dataset = await request(example ? "/api/datasets/example" : "/api/datasets", { method: "POST", body: example ? undefined : form });
    state.datasets.unshift(dataset); state.datasetId = dataset.id;
    renderDatasets(); $("global-error").hidden = true;
    showToast(`${example ? "示例数据已添加" : "上传成功"} · ${number(dataset.rows)} 行记录`);
  } catch (error) { showError(error); }
  finally { setUploading(false); $("file-input").value = ""; }
}
$("pipeline-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.datasetId || state.submitting || !event.currentTarget.reportValidity()) return;
  state.submitting = true; updateStartState();
  try {
    const run = await request("/api/runs", { method: "POST", body: {
      dataset_id: state.datasetId, provider: $("provider").value, trainer: $("trainer").value,
      confidence: Number($("confidence").value), concurrency: Number($("concurrency").value),
      max_requests: Number($("max-requests").value), auto_train: $("auto-train").checked,
    } });
    upsertRun(run); state.runId = run.id; renderRunList(); renderRun(); schedulePoll();
    $("global-error").hidden = true; showToast("工作流已创建，正在开始处理。");
    $("run-section").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (error) { showError(error); }
  finally { state.submitting = false; updateStartState(); }
});
async function runAction(action) {
  if (!state.runId || state.actionPending) return;
  const runId = state.runId;
  state.actionPending = true; renderRun();
  try {
    const run = await request(`/api/runs/${encodeURIComponent(runId)}/${action}`, { method: "POST" });
    upsertRun(run); if (state.runId === runId) state.runId = run.id;
    renderRunList(); renderRun(); schedulePoll();
    showToast(action === "cancel" ? "已请求停止，任务会在当前步骤结束时退出。" : "工作流已重新启动。");
  } catch (error) { showError(error); }
  finally { state.actionPending = false; renderRun(); }
}
$("cancel-button").addEventListener("click", () => runAction("cancel"));
$("retry-button").addEventListener("click", () => runAction("retry"));
$("confidence").addEventListener("input", () => { $("confidence-value").textContent = Number($("confidence").value).toFixed(2); });
for (const id of ["provider", "trainer", "auto-train"]) $(id).addEventListener("change", updateModeNotice);
$("dataset-select").addEventListener("change", (event) => { state.datasetId = event.target.value; renderDatasetPreview(); updateStartState(); });
$("file-input").addEventListener("change", (event) => uploadDataset(event.target.files[0]));
$("example-button").addEventListener("click", () => uploadDataset(null, true));
$("refresh-button").addEventListener("click", loadWorkspace);
let dragDepth = 0;
$("drop-zone").addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth++; if (!state.uploading) $("drop-zone").classList.add("drag-over"); });
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; });
$("drop-zone").addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop-zone").classList.remove("drag-over"); });
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("drop-zone").classList.remove("drag-over"); if (event.dataTransfer?.files.length > 1) showToast("每次上传一个数据集，已选择第一个文件。"); uploadDataset(event.dataTransfer?.files[0]); });
$("settings-button").addEventListener("click", () => { $("api-token").value = state.token; $("settings-dialog").showModal(); });
$("close-settings").addEventListener("click", () => $("settings-dialog").close());
$("settings-dialog").addEventListener("click", (event) => { if (event.target === $("settings-dialog")) { const bounds = event.target.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.target.close(); } });
$("settings-form").addEventListener("submit", (event) => {
  event.preventDefault(); state.token = $("api-token").value.trim();
  if (state.token) sessionStorage.setItem("jev_api_token", state.token); else sessionStorage.removeItem("jev_api_token");
  $("settings-dialog").close(); loadWorkspace();
});
for (const link of document.querySelectorAll("nav a")) link.addEventListener("click", () => {
  for (const item of document.querySelectorAll("nav a")) item.classList.toggle("active", item === link);
});
document.addEventListener("visibilitychange", () => { if (!document.hidden && state.runs.some(isActive)) schedulePoll(0); });
loadWorkspace();
