"use strict";

(() => {
  const domains = { general: "General", finance: "Finance", code: "Code", enterprise: "Enterprise", legal: "Legal", medical: "Medical" };
  const evidence = {
    computed: { label: "Computed", description: "Calculated deterministically from the input fields you supply. This calculation does not establish that a reference or source is correct." },
    external: { label: "External tests", description: "Aggregates test results produced elsewhere and supplied in the record. This evaluator does not run tests, tools, or generated code." },
    human: { label: "Human labels", description: "Aggregates labels or ratings supplied by annotators or reviewers. This evaluator does not create expert judgments or independently validate them." },
  };
  const methods = {
    exact_match: "Compares prediction and reference after NFC Unicode and whitespace normalization. Matching text scores 1; a mismatch scores 0. Case and punctuation remain significant; an empty reference is invalid.",
    token_f1: "Uses case-sensitive whitespace tokens after NFC and whitespace normalization: 2 × multiset token overlap / total prediction and reference token count. It measures surface overlap, not semantic or factual correctness; an empty reference is invalid.",
    numeric_tolerance: "Scores 1 when absolute prediction error is at most max(absolute_tolerance, relative_tolerance × absolute reference). If unit fields are declared, their strings must match exactly; no unit conversion is performed.",
    set_precision: "Unique exact-string intersection / predicted set size. No source retrieval or semantic matching is performed. An empty prediction against a nonempty reference scores 0; two empty sets are not applicable.",
    set_recall: "Unique exact-string intersection / reference set size. No source retrieval or semantic matching is performed. An empty reference set is not applicable.",
    ratio: "Supplied numerator / denominator, calculated per record. Counts must be nonnegative integers with numerator ≤ denominator. A zero denominator is not applicable; counts are not independently measured here.",
    boolean: "Maps the supplied boolean value to 1 for true and 0 for false. It aggregates an existing decision; it does not make that decision.",
    rubric_score: "Normalizes a supplied integer rating as (rating − min) / (max − min). Out-of-range ratings are invalid. An annotator or external process must provide the rating first.",
    json_valid: "Scores whether supplied text parses as JSON and matches the configured top-level type (object by default). Duplicate object keys and non-finite constants are rejected. This does not validate an application schema or content correctness.",
  };
  const $ = id => document.getElementById(id);
  const cache = new Map();
  let currentDomain = "general", currentPack = null, requestVersion = 0, abortController;
  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = String(text);
    return item;
  }
  function humanize(value) { return String(value).replaceAll("_", " "); }
  function validatePack(pack, domain) {
    if (!pack || pack.schema_version !== 1 || pack.domain !== domain || typeof pack.name !== "string" || !Array.isArray(pack.metrics) || !pack.field_schema || typeof pack.field_schema !== "object" || Array.isArray(pack.field_schema)) throw new Error("The metric pack has an unsupported or incomplete schema.");
    const ids = new Set();
    for (const metric of pack.metrics) {
      if (!metric || typeof metric.id !== "string" || ids.has(metric.id) || typeof metric.name !== "string" || typeof metric.operator !== "string" || !Object.hasOwn(evidence, metric.evidence) || !metric.fields || typeof metric.fields !== "object" || Array.isArray(metric.fields)) throw new Error("A metric definition is incomplete or has an unsupported evidence type.");
      ids.add(metric.id);
      for (const path of Object.values(metric.fields)) if (typeof path !== "string" || !Object.hasOwn(pack.field_schema, path)) throw new Error("A required input field is missing from the pack's field schema.");
    }
    return pack;
  }
  function updateCommands(domain) {
    $("metric-init-code").textContent = `jev-dataops init-metrics \\\n  --domain ${domain} \\\n  --output ./${domain}-metrics`;
    $("metric-evaluate-code").textContent = `jev-dataops evaluate-metrics \\\n  --input ./${domain}-metrics/example.jsonl \\\n  --pack ./${domain}-metrics/metrics.json \\\n  --output ./${domain}-results`;
  }
  function appendDetail(container, title, text, className = "") {
    const group = node("div", `metric-detail-group ${className}`.trim());
    group.append(node("h4", "", title), node("p", "", text));
    container.append(group);
  }
  function createMetricCard(metric, pack) {
    const article = node("article", "metric-definition-card");
    const top = node("div", "metric-card-top");
    top.append(node("code", "metric-id", metric.id), node("span", `evidence-badge ${metric.evidence}`, evidence[metric.evidence].label));
    article.append(top, node("h3", "", metric.name), node("p", "metric-description", metric.description || "No description supplied."));
    const meta = node("dl", "metric-card-meta");
    for (const [label, value] of [["Method", metric.operator], ["Unit", metric.unit || "Not specified"], ["Direction", humanize(metric.direction || "not specified")]]) {
      const pair = node("div"); pair.append(node("dt", "", label), node("dd", "", value)); meta.append(pair);
    }
    article.append(meta);
    const details = node("details", "metric-detail");
    const summary = node("summary", "", "Inputs, computation & limits");
    const symbol = node("span", "", "+"); symbol.setAttribute("aria-hidden", "true"); summary.append(symbol); details.append(summary);
    const content = node("div", "metric-detail-content");
    appendDetail(content, "Computation", methods[metric.operator] || `Uses the ${metric.operator} operator defined by the offline evaluator.`);
    appendDetail(content, "Aggregation", `${humanize(metric.aggregation || "not specified")} across evaluated records. Coverage is evaluated / eligible records; not-applicable records are excluded from the eligible denominator. Missing and invalid inputs reduce coverage rather than becoming zero-quality results.`);
    const parameters = metric.parameters && typeof metric.parameters === "object" ? Object.entries(metric.parameters) : [];
    if (parameters.length) {
      const group = node("div", "metric-detail-group"); group.append(node("h4", "", "Configured parameters"), node("pre", "metric-parameters", JSON.stringify(metric.parameters, null, 2))); content.append(group);
    }
    const fieldGroup = node("div", "metric-detail-group"); fieldGroup.append(node("h4", "", "Required input fields"));
    const fields = node("ul", "metric-input-fields");
    for (const [role, path] of Object.entries(metric.fields)) {
      const definition = pack.field_schema[path];
      const field = node("li");
      const title = node("div", "input-field-title"); title.append(node("code", "", path), node("span", "field-type", definition.type || "Not specified"));
      field.append(title, node("p", "input-field-role", `${humanize(role)} · ${definition.description || "No field description supplied."}`)); fields.append(field);
    }
    fieldGroup.append(fields); content.append(fieldGroup);
    appendDetail(content, "Evidence & provenance", evidence[metric.evidence].description, `evidence-explanation ${metric.evidence}`);
    appendDetail(content, "Interpretation limits", metric.limitations || "No metric-specific limits were supplied. Review the pack-level limitations.");
    let target = "No target configured (null). Calibrate a task-specific threshold and minimum coverage before using a pass decision.";
    if (metric.target && typeof metric.target === "object") target = `Pack-configured target: minimum ${metric.target.minimum ?? "not specified"}; minimum coverage ${metric.target.minimum_coverage ?? "not specified"}. This is a configuration choice, not an industry-approved standard.`;
    appendDetail(content, "Target status", target);
    details.append(content); article.append(details);
    return article;
  }
  function renderMetricGrid() {
    if (!currentPack) return;
    const query = $("metric-search").value.trim().toLowerCase();
    const filter = $("metric-evidence-filter").value;
    const matches = currentPack.metrics.filter(metric => {
      const searchable = [metric.id, metric.name, metric.description, metric.operator, ...Object.values(metric.fields)].join(" ").toLowerCase();
      return (filter === "all" || metric.evidence === filter) && (!query || searchable.includes(query));
    });
    const fragment = document.createDocumentFragment();
    for (const metric of matches) fragment.append(createMetricCard(metric, currentPack));
    $("metric-grid").replaceChildren(fragment);
    $("metric-no-results").hidden = matches.length !== 0;
    $("metric-filter-status").textContent = `Showing ${matches.length} of ${currentPack.metrics.length} metric definitions.`;
  }
  function renderFields(pack) {
    const fragment = document.createDocumentFragment();
    for (const [path, definition] of Object.entries(pack.field_schema)) {
      const users = pack.metrics.filter(metric => Object.values(metric.fields).includes(path));
      const row = node("tr");
      const name = node("th"); name.scope = "row"; name.append(node("code", "", path));
      const type = node("td"); type.append(node("span", "field-type", definition.type || "Not specified"));
      row.append(name, type, node("td", "", definition.description || "No description supplied."), node("td", "field-users", users.length ? users.map(metric => metric.name).join(" · ") : "Record context / provenance"));
      fragment.append(row);
    }
    $("field-schema-body").replaceChildren(fragment);
  }
  function renderPack(pack, records, domain) {
    currentPack = pack;
    $("pack-domain-label").textContent = `${domains[domain].toUpperCase()} / METRIC PACK`;
    $("pack-version").textContent = `v${pack.version ?? "1"} · schema ${pack.schema_version}`;
    $("pack-name").textContent = pack.name;
    $("pack-description").textContent = pack.description || "";
    $("metric-total").textContent = pack.metrics.length;
    $("computed-total").textContent = pack.metrics.filter(metric => metric.evidence === "computed").length;
    $("supplied-total").textContent = pack.metrics.filter(metric => metric.evidence !== "computed").length;
    $("field-total").textContent = Object.keys(pack.field_schema).length;
    $("download-pack").href = `/assets/metric-packs/${domain}.json`;
    $("download-pack").download = `${domain}-metrics.json`;
    $("download-example").href = `/assets/metric-examples/${domain}.jsonl`;
    $("download-example").download = `${domain}-example.jsonl`;
    $("example-filename").textContent = `${domain}.jsonl · first record`;
    $("example-count").textContent = `${records.length} synthetic example ${records.length === 1 ? "record" : "records"} in this download`;
    $("metric-example-code").textContent = JSON.stringify(records[0], null, 2);
    $("metric-search").value = ""; $("metric-evidence-filter").value = "all";
    $("pack-limitations").replaceChildren(...(Array.isArray(pack.limitations) ? pack.limitations.map(item => node("li", "", item)) : []));
    renderFields(pack); renderMetricGrid();
    $("metric-pack-content").hidden = false;
  }
  async function loadDomain(domain, updateURL = true) {
    if (!Object.hasOwn(domains, domain)) domain = "general";
    currentDomain = domain;
    const version = ++requestVersion;
    abortController?.abort(); abortController = new AbortController();
    $("metric-pack-content").hidden = true; $("metric-error").hidden = true; $("metric-loading").hidden = false;
    $("metric-loading").querySelector("p").textContent = `Loading the ${domains[domain]} metric pack…`;
    $("catalog").setAttribute("aria-busy", "true");
    for (const button of document.querySelectorAll("[data-domain]")) button.setAttribute("aria-pressed", String(button.dataset.domain === domain));
    updateCommands(domain);
    if (updateURL) { const url = new URL(location.href); url.searchParams.set("domain", domain); history.replaceState(null, "", url.pathname + url.search + url.hash); }
    try {
      let result = cache.get(domain);
      if (!result) {
        const signal = abortController.signal;
        const [packResponse, exampleResponse] = await Promise.all([
          fetch(`/assets/metric-packs/${domain}.json`, { signal, credentials: "same-origin" }),
          fetch(`/assets/metric-examples/${domain}.jsonl`, { signal, credentials: "same-origin" }),
        ]);
        if (!packResponse.ok || !exampleResponse.ok) throw new Error(`The static pack or example file is unavailable (HTTP ${!packResponse.ok ? packResponse.status : exampleResponse.status}).`);
        const pack = validatePack(await packResponse.json(), domain);
        const lines = (await exampleResponse.text()).trim().split(/\r?\n/).filter(line => line.trim());
        const records = lines.map(line => JSON.parse(line));
        if (!records.length || records.some(record => !record || typeof record !== "object" || Array.isArray(record))) throw new Error("The example JSONL file is empty or contains an invalid record.");
        result = { pack, records }; cache.set(domain, result);
      }
      if (version !== requestVersion) return;
      renderPack(result.pack, result.records, domain);
    } catch (error) {
      if (version !== requestVersion || error.name === "AbortError") return;
      currentPack = null;
      $("metric-error-message").textContent = error instanceof SyntaxError ? "A static pack or example file could not be parsed. Please retry or inspect the source repository." : `${error.message} No metric definitions have been substituted.`;
      $("metric-error").hidden = false;
    } finally {
      if (version === requestVersion) { $("metric-loading").hidden = true; $("catalog").setAttribute("aria-busy", "false"); }
    }
  }
  const domainButtons = [...document.querySelectorAll("[data-domain]")];
  domainButtons.forEach((button, index) => {
    button.addEventListener("click", () => loadDomain(button.dataset.domain));
    button.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % domainButtons.length;
      else if (event.key === "ArrowLeft") next = (index + domainButtons.length - 1) % domainButtons.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = domainButtons.length - 1;
      else return;
      event.preventDefault(); domainButtons[next].focus(); loadDomain(domainButtons[next].dataset.domain);
    });
  });
  $("metric-retry").addEventListener("click", () => loadDomain(currentDomain, false));
  $("metric-search").addEventListener("input", renderMetricGrid);
  $("metric-evidence-filter").addEventListener("change", renderMetricGrid);
  $("metric-reset-filter").addEventListener("click", () => { $("metric-search").value = ""; $("metric-evidence-filter").value = "all"; renderMetricGrid(); $("metric-search").focus(); });
  loadDomain(new URLSearchParams(location.search).get("domain") || "general", false);
})();
