import { useCallback, useEffect, useState } from "react";

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.error || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function StatusMark({ online }) {
  return <span className={`status-mark ${online ? "online" : "offline"}`} />;
}

function AgentRun({ health }) {
  const [request, setRequest] = useState("");
  const [active, setActive] = useState(null);
  const [pollError, setPollError] = useState("");
  const [approvalToken, setApprovalToken] = useState("");
  const [approving, setApproving] = useState(false);

  const refresh = useCallback(async () => {
    if (!active?.task_id) return;
    try {
      const task = await api(`/tasks/${encodeURIComponent(active.task_id)}`);
      if (task.status === "SUCCESS") {
        setActive((current) => ({ ...current, outcome: task.result, status: "SUCCESS" }));
        setPollError("");
      } else if (task.status === "FAILURE") {
        setActive((current) => ({ ...current, status: "FAILURE" }));
        setPollError(task.error || "The background task failed. Check the API and worker logs.");
      } else {
        setActive((current) => ({ ...current, status: task.status }));
      }
    } catch (error) {
      setPollError(error.message);
    }
  }, [active?.task_id]);

  useEffect(() => {
    if (!active?.task_id || ["SUCCESS", "FAILURE"].includes(active.status)) return undefined;
    const interval = window.setInterval(refresh, 1500);
    refresh();
    return () => window.clearInterval(interval);
  }, [active?.task_id, active?.status, refresh]);

  async function submit(event) {
    event.preventDefault();
    if (!request.trim()) return;
    setPollError("");
    try {
      const result = await api("/agents/run", {
        method: "POST",
        body: JSON.stringify({ request: request.trim() }),
      });
      setActive({
        task_id: result.task_id,
        thread_id: result.thread_id,
        status: "PENDING",
        submittedAt: Date.now(),
      });
    } catch (error) {
      setPollError(error.message);
    }
  }

  async function decide(approved) {
    if (!active?.thread_id) return;
    setApproving(true);
    setPollError("");
    try {
      const result = await api(
        `/agents/${encodeURIComponent(active.thread_id)}/approval`,
        {
          method: "POST",
          headers: approvalToken ? { "X-Approval-Token": approvalToken } : {},
          body: JSON.stringify({ approved }),
        },
      );
      setApprovalToken("");
      setActive({
        task_id: result.task_id,
        thread_id: result.thread_id,
        status: "PENDING",
        submittedAt: Date.now(),
      });
    } catch (error) {
      setPollError(error.message);
    } finally {
      setApproving(false);
    }
  }

  const outcome = active?.outcome;
  const approval = outcome?.status === "awaiting_approval"
    ? outcome.interrupts?.[0]
    : null;

  return (
    <div className="work-grid">
      <section className="panel request-panel">
        <div className="panel-heading">
          <div><span className="kicker">01 / EXPERIMENT</span><h2>Send a request</h2></div>
          <span className="tag">OmniRoute · {health?.llm_model || "configured model"}</span>
        </div>
        <p className="muted">
          Try a general question, search your indexed sample documents, or ask to
          delete a document to test the approval checkpoint.
        </p>
        <form onSubmit={submit}>
          <label htmlFor="request">What should the agent do?</label>
          <textarea
            id="request"
            rows="6"
            maxLength="20000"
            value={request}
            onChange={(event) => setRequest(event.target.value)}
            placeholder="Search our internal documents for the data-retention policy and summarize the key points."
          />
          <div className="prompt-samples">
            <button type="button" onClick={() => setRequest("Search our internal documents for the data-retention policy and summarize the key points.")}>Search example</button>
            <button type="button" onClick={() => setRequest("Delete the internal document with ID demo-policy.")}>Approval example</button>
          </div>
          <button
            className="button primary"
            type="submit"
            disabled={
              !request.trim()
              || health?.broker_connected === false
              || health?.redis_search_available === false
              || health?.worker_connected === false
            }
          >
            Run agent <span aria-hidden="true">↗</span>
          </button>
        </form>
        <p className="footnote">Use non-sensitive test text. The agent only has the tools listed on this page.</p>
      </section>

      <section className="panel trace-panel">
        <div className="panel-heading">
          <div><span className="kicker">02 / OBSERVE</span><h2>Run trace</h2></div>
          <span className={`tag ${active ? "live-tag" : ""}`}>
            {active ? active.status : "No run yet"}
          </span>
        </div>
        {!active && !pollError && <div className="empty"><span>◌</span><strong>Nothing running yet</strong><p>Submit a request to see the plan and result here.</p></div>}
        {!active && pollError && (
          <div className="trace-content">
            <div className="error-box"><strong>Could not start the agent run.</strong><p>{pollError}</p><p>Check that Redis Stack, FastAPI, and the Celery worker are running, then try again.</p></div>
          </div>
        )}
        {active && (
          <div className="trace-content">
            <div className="ids">
              <span>THREAD <code>{active.thread_id}</code></span>
              <span>TASK <code>{active.task_id}</code></span>
            </div>
            {!outcome && !pollError && (
              <div>
                <p className="running"><span className="spinner" /> Waiting for the worker…</p>
                {active.status === "PENDING" && active.submittedAt && Date.now() - active.submittedAt > 12000 && (
                  <p className="worker-hint">Still queued after 12 seconds. The Celery worker may not be running; check the worker terminal started by the launcher.</p>
                )}
              </div>
            )}
            {outcome?.plan && (
              <div className="plan-box">
                <h3>Agent plan</h3>
                <ol>{outcome.plan.steps?.map((step, index) => <li key={`${index}-${step}`}>{step}</li>)}</ol>
                {outcome.plan.rationale && <p>{outcome.plan.rationale}</p>}
              </div>
            )}
            {approval && (
              <div className="approval-box">
                <h3>Human approval required</h3>
                <p>The agent will not execute this action until you decide.</p>
                {approval.tool_calls?.map((call, index) => (
                  <pre key={`${call.name}-${index}`}>{call.name}{"\n"}{JSON.stringify(call.arguments, null, 2)}</pre>
                ))}
                {health?.approval_token_required && (
                  <div className="approval-token">
                    <label htmlFor="approval-token">Local approval key</label>
                    <input
                      id="approval-token"
                      type="password"
                      autoComplete="off"
                      value={approvalToken}
                      onChange={(event) => setApprovalToken(event.target.value)}
                      placeholder="APPROVAL_API_KEY from .env"
                    />
                  </div>
                )}
                <div className="button-row">
                  <button className="button primary" onClick={() => decide(true)} disabled={approving}>Approve action</button>
                  <button className="button secondary" onClick={() => decide(false)} disabled={approving}>Reject action</button>
                </div>
              </div>
            )}
            {outcome?.status === "completed" && (
              <div className="answer-box"><span className="kicker">AGENT RESPONSE</span><p>{outcome.response}</p></div>
            )}
            {pollError && <div className="error-box">{pollError}</div>}
            {active.status === "FAILURE" && !pollError && <div className="error-box">The task failed. Check the API and worker logs.</div>}
            {["SUCCESS", "FAILURE"].includes(active.status) && (
              <button className="text-button" onClick={() => { setActive(null); setPollError(""); }}>Clear run</button>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function Knowledge() {
  const [documentId, setDocumentId] = useState("demo-policy");
  const [content, setContent] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function indexDocument(event) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      const response = await api("/documents", {
        method: "PUT",
        body: JSON.stringify({
          documents: [{ id: documentId.trim(), content: content.trim(), metadata: {} }],
        }),
      });
      setMessage(`Indexed ${response.indexed} document. Try searching for it.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function search(event) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      setResults(await api("/search", {
        method: "POST",
        body: JSON.stringify({ query: query.trim(), top_k: 5 }),
      }));
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel knowledge-panel">
      <div className="panel-heading">
        <div><span className="kicker">03 / RAG SANDBOX</span><h2>Documents & hybrid search</h2></div>
        <span className="tag">Local PostgreSQL + pgvector</span>
      </div>
      <p className="muted">
        Index a short, non-sensitive sample. The local embedding model downloads
        automatically the first time you index or search.
      </p>
      <div className="knowledge-grid">
        <form onSubmit={indexDocument}>
          <label htmlFor="document-id">Document ID</label>
          <input id="document-id" required value={documentId} onChange={(event) => setDocumentId(event.target.value)} />
          <label htmlFor="document-content">Sample document text</label>
          <textarea id="document-content" required rows="5" value={content} onChange={(event) => setContent(event.target.value)} placeholder="Paste a short internal-policy example or test text." />
          <button className="button secondary" type="submit" disabled={busy || !content.trim()}>Index document</button>
        </form>
        <div>
          <form onSubmit={search}>
            <label htmlFor="query">Search query</label>
            <div className="search-row">
              <input id="query" required value={query} onChange={(event) => setQuery(event.target.value)} placeholder="What are you looking for?" />
              <button className="button secondary" type="submit" disabled={busy || !query.trim()}>Search</button>
            </div>
          </form>
          <div className="results">
            {results?.length === 0 && <p className="muted">No matches yet. Index a sample document first.</p>}
            {results?.map((result) => (
              <article className="result" key={result.id}>
                <div><strong>{result.id}</strong><span>score {result.score.toFixed(4)}</span></div>
                <p>{result.content}</p>
              </article>
            ))}
            {results === null && <p className="muted">Your search results appear here.</p>}
          </div>
        </div>
      </div>
      {message && <p className={`notice ${message.includes("failed") || message.includes("Could not") ? "notice-error" : ""}`}>{message}</p>}
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    let active = true;
    async function checkApi() {
      try {
        const status = await api("/health");
        if (active) {
          setHealth(status);
          setHealthError("");
        }
      } catch (error) {
        if (active) {
          setHealth(null);
          setHealthError(error.message);
        }
      }
    }
    checkApi();
    const interval = window.setInterval(checkApi, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const runtimeIssue = health && !health.broker_connected
    ? "Redis is not connected. Start Docker Desktop, then run .\\Start-AgentLab.ps1."
    : health && !health.redis_search_available
      ? "Redis is running without its Search module. From the project folder run: docker compose up -d --force-recreate redis."
      : health && !health.worker_connected
      ? "The Celery worker is not responding. Keep the worker window open and restart it with .\\Start-AgentLab.ps1."
      : "";
  const connectionLabel = healthError
    ? "API offline"
    : runtimeIssue
      ? `API online · ${!health?.broker_connected ? "Redis offline" : !health?.redis_search_available ? "Redis Stack needed" : "worker offline"}`
      : health
        ? `API online · ${health.llm_provider}`
        : "Connecting to API";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">AO</span>
          <span>AGENT <b>LAB</b><small>ORCHESTRATION SYSTEM</small></span>
        </a>
        <div className="connection">
          <StatusMark online={Boolean(health && !runtimeIssue)} />
          <span>{connectionLabel}</span>
        </div>
      </header>
      <main>
        <section className="hero">
          <span className="kicker">LOCAL AGENT WORKBENCH</span>
          <h1>From request to <em>result.</em></h1>
          <p>Run a real agent task, understand the tool it chooses, and inspect what happened.</p>
          {healthError && <div className="error-box api-error"><strong>FastAPI is not reachable.</strong> Start the API and Celery worker with <code>.\Start-AgentLab.ps1</code>. {healthError}</div>}
          {runtimeIssue && <div className="error-box api-error"><strong>Agent services are not ready.</strong> {runtimeIssue}</div>}
          {health && !health.llm_credential_configured && <div className="error-box api-error">Add your OmniRoute key to <code>LLM_API_KEY</code> in your private <code>.env</code> file, then restart the API.</div>}
          {health && <div className="flow-strip">
            <div><span>01</span><b>Plan</b><small>Break down the request</small></div>
            <i>→</i><div><span>02</span><b>Act</b><small>Call a registered tool</small></div>
            <i>→</i><div><span>03</span><b>Approve</b><small>Pause before deletion</small></div>
            <i>→</i><div><span>04</span><b>Reflect</b><small>Explain the outcome</small></div>
          </div>}
        </section>
        {healthError ? (
          <div className="panel offline-panel">
            <h2>Start the local services</h2>
            <p>From the project directory, run <code>.\Start-AgentLab.ps1</code> in PowerShell, then refresh this page.</p>
          </div>
        ) : (
          <>
            <AgentRun health={health} />
            <Knowledge />
          </>
        )}
        <section className="capabilities">
          <div><span className="cap-icon">⌕</span><div><strong>Search documents</strong><p>Hybrid keyword + meaning search over documents you index.</p></div></div>
          <div><span className="cap-icon amber">◈</span><div><strong>Delete with approval</strong><p>Deletion waits until you approve the exact document ID.</p></div></div>
          <div><span className="cap-icon">↻</span><div><strong>No hidden tools</strong><p>No shell, web browsing, or access to files on your computer.</p></div></div>
        </section>
      </main>
      <footer><span>LOCAL DEVELOPMENT ONLY</span><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">FastAPI docs ↗</a></footer>
    </div>
  );
}
