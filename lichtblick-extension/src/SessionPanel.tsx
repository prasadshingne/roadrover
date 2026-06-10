import { PanelExtensionContext } from "@lichtblick/suite";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

// ── Types ─────────────────────────────────────────────────────────────────────

type Chunk = {
  name: string;
  processed: boolean;
  has_scenario: boolean;
  raw_mcap: boolean;
  processed_mcap: boolean;
  raw_mcap_url: string | null;
  processed_mcap_url: string | null;
  scenario_xosc: string | null;
};

type Session = {
  id: string;
  name: string;
  map_ready: boolean;
  chunked: boolean;
  processed: boolean;
  has_scenario: boolean;
  chunks: Chunk[];
};

type Job = {
  job_id: string;
  label: string;
  status: "running" | "done" | "error";
  elapsed: number;
};

type PanelState = { serverUrl: string };

// ── Styles ────────────────────────────────────────────────────────────────────

const S = {
  root: {
    fontFamily: "monospace",
    fontSize: 12,
    background: "#1a1a1a",
    color: "#e0e0e0",
    height: "100%",
    overflowY: "auto" as const,
    display: "flex",
    flexDirection: "column" as const,
  },
  toolbar: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    background: "#222",
    borderBottom: "1px solid #333",
    flexShrink: 0,
  },
  urlInput: {
    flex: 1,
    background: "#111",
    border: "1px solid #444",
    color: "#ccc",
    padding: "2px 6px",
    borderRadius: 3,
    fontSize: 11,
    fontFamily: "monospace",
  },
  body: { padding: 8, flex: 1, overflowY: "auto" as const },
  session: {
    background: "#252525",
    border: "1px solid #333",
    borderRadius: 4,
    marginBottom: 6,
  },
  sessionHeader: {
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap" as const,
    gap: 5,
    padding: "6px 8px",
    cursor: "pointer",
  },
  sname: { flex: 1, fontWeight: "bold" as const, fontSize: 12 },
  chunks: { padding: "4px 8px 8px 20px", borderTop: "1px solid #2a2a2a" },
  chunk: {
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap" as const,
    gap: 4,
    padding: "4px 0",
    borderBottom: "1px solid #282828",
  },
  chunkName: { width: 130, color: "#999", flexShrink: 0 },
  jobRow: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "3px 8px",
    background: "#222",
    borderRadius: 3,
    marginBottom: 3,
  },
  logBox: {
    fontSize: 10,
    color: "#888",
    background: "#111",
    padding: "5px 8px",
    maxHeight: 140,
    overflowY: "auto" as const,
    whiteSpace: "pre" as const,
    borderRadius: 3,
    marginBottom: 3,
  },
  toast: {
    position: "fixed" as const,
    bottom: 12,
    right: 12,
    background: "#333",
    color: "#7ec8e3",
    padding: "6px 12px",
    borderRadius: 4,
    fontSize: 11,
    zIndex: 999,
    maxWidth: 320,
  },
  sectionTitle: { fontSize: 11, color: "#666", marginBottom: 4, marginTop: 10 },
} as const;

// ── Badge component ───────────────────────────────────────────────────────────

function Badge({ ok, label }: { ok: boolean; label: string }) {
  if (!ok) return null;
  return (
    <span
      style={{
        padding: "1px 5px",
        borderRadius: 3,
        fontSize: 10,
        background: ok ? "#1a3d24" : "#3a1515",
        color: ok ? "#6fcf97" : "#eb5757",
      }}
    >
      {label}
    </span>
  );
}

function StatusBadge({ ok, trueLabel, falseLabel }: { ok: boolean; trueLabel: string; falseLabel: string }) {
  return (
    <span
      style={{
        padding: "1px 5px",
        borderRadius: 3,
        fontSize: 10,
        background: ok ? "#1a3d24" : "#3a1515",
        color: ok ? "#6fcf97" : "#eb5757",
      }}
    >
      {ok ? trueLabel : falseLabel}
    </span>
  );
}

// ── Btn component ─────────────────────────────────────────────────────────────

function Btn({
  children,
  onClick,
  primary,
  disabled,
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  primary?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      style={{
        padding: "2px 7px",
        border: `1px solid ${primary ? "#7ec8e3" : "#555"}`,
        background: "#2a2a2a",
        color: primary ? "#7ec8e3" : "#ccc",
        borderRadius: 3,
        cursor: disabled ? "default" : "pointer",
        fontSize: 11,
        fontFamily: "monospace",
        opacity: disabled ? 0.4 : 1,
      }}
    >
      {children}
    </button>
  );
}

// ── Main panel component ──────────────────────────────────────────────────────

function SessionPanel({
  serverUrl: initialUrl,
  onUrlChange,
}: {
  serverUrl: string;
  onUrlChange: (url: string) => void;
}) {
  const [serverUrl, setServerUrl] = useState(initialUrl);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [openLogs, setOpenLogs] = useState<Record<string, string[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  }, []);

  const api = useCallback(
    async (method: string, path: string, qs?: Record<string, string>) => {
      const url =
        serverUrl +
        path +
        (qs ? "?" + new URLSearchParams(qs).toString() : "");
      const r = await fetch(url, { method });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(text);
      }
      return r.json();
    },
    [serverUrl],
  );

  const loadSessions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api("GET", "/api/sessions");
      setSessions(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [api]);

  const loadJobs = useCallback(async () => {
    try {
      const data = await api("GET", "/api/jobs");
      setJobs(data);
    } catch (_) {}
  }, [api]);

  useEffect(() => {
    loadSessions();
    loadJobs();
    timerRef.current = setInterval(loadJobs, 4000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [loadSessions, loadJobs]);

  const postAction = useCallback(
    async (path: string, label: string, qs?: Record<string, string>) => {
      try {
        const r = await api("POST", path, qs);
        showToast(`${label} started (job ${r.job_id})`);
        setTimeout(loadJobs, 800);
      } catch (e) {
        showToast(`Error: ${e}`);
      }
    },
    [api, loadJobs, showToast],
  );

  const copyMcapUrl = useCallback(
    (relUrl: string) => {
      const full = serverUrl + relUrl;
      navigator.clipboard.writeText(full).then(() => {
        showToast(
          "Copied! In Lichtblick: File > Open > Remote file > paste URL",
        );
      });
    },
    [serverUrl, showToast],
  );

  const toggleLog = useCallback(
    async (jobId: string) => {
      if (openLogs[jobId]) {
        setOpenLogs((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
        return;
      }
      try {
        const j = await api("GET", `/api/jobs/${jobId}`);
        setOpenLogs((prev) => ({ ...prev, [jobId]: j.log_tail }));
      } catch (_) {}
    },
    [api, openLogs],
  );

  const handleUrlCommit = useCallback(
    (url: string) => {
      setServerUrl(url);
      onUrlChange(url);
    },
    [onUrlChange],
  );

  const statusColor = { running: "#f2c94c", done: "#6fcf97", error: "#eb5757" };

  return (
    <div style={S.root}>
      {/* Toolbar */}
      <div style={S.toolbar}>
        <span style={{ color: "#7ec8e3", fontWeight: "bold" }}>🗺 Roadrover</span>
        <input
          style={S.urlInput}
          value={serverUrl}
          onChange={(e) => setServerUrl(e.target.value)}
          onBlur={(e) => handleUrlCommit(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleUrlCommit(serverUrl)}
          placeholder="http://localhost:8765"
          title="Studio server URL"
        />
        <Btn onClick={loadSessions} disabled={loading} title="Reload sessions">
          {loading ? "…" : "↺"}
        </Btn>
      </div>

      <div style={S.body}>
        {/* Error state */}
        {error && (
          <div style={{ color: "#eb5757", padding: "6px 0", fontSize: 11 }}>
            Could not reach server: {error}
            <br />
            <span style={{ color: "#888" }}>
              Start: <code>python3 studio_server.py</code> in a ROS 2 terminal
            </span>
          </div>
        )}

        {/* Sessions */}
        {sessions.length === 0 && !error && !loading && (
          <div style={{ color: "#555", padding: "8px 0" }}>
            No sessions found in ~/roadrover_bags
          </div>
        )}
        {sessions.map((s) => (
          <div key={s.id} style={S.session}>
            <div
              style={S.sessionHeader}
              onClick={() =>
                setExpanded((prev) => ({ ...prev, [s.id]: !prev[s.id] }))
              }
            >
              <span style={S.sname}>{s.name}</span>
              <StatusBadge ok={s.map_ready} trueLabel="map ✓" falseLabel="no map" />
              <StatusBadge ok={s.processed} trueLabel="processed" falseLabel="raw" />
              <Badge ok={s.has_scenario} label="scenario" />
              <span
                onClick={(e) => e.stopPropagation()}
                style={{ display: "flex", gap: 4 }}
              >
                <Btn
                  primary
                  title="Run full pipeline: split → YOLO → lanes → EKF"
                  onClick={() => postAction(`/api/sessions/${s.id}/process`, "Process")}
                >
                  Process
                </Btn>
                <Btn
                  title="Extract OpenSCENARIO from all processed chunks"
                  onClick={() => postAction(`/api/sessions/${s.id}/extract`, "Extract")}
                >
                  Extract scenario
                </Btn>
                <Btn
                  title="Convert db3 bags to MCAP so Lichtblick can open via URL"
                  onClick={() => postAction(`/api/sessions/${s.id}/convert-mcap`, "Convert MCAP")}
                >
                  → MCAP
                </Btn>
              </span>
            </div>

            {/* Chunks */}
            {expanded[s.id] && (
              <div style={S.chunks}>
                {s.chunks.length === 0 && (
                  <span style={{ color: "#555" }}>No chunks found.</span>
                )}
                {s.chunks.map((c) => (
                  <div key={c.name} style={S.chunk}>
                    <span style={S.chunkName}>{c.name}</span>
                    <StatusBadge ok={c.processed} trueLabel="proc" falseLabel="raw" />
                    <Badge ok={c.has_scenario} label="scenario" />
                    <Badge ok={c.raw_mcap} label="raw mcap" />
                    <Badge ok={c.processed_mcap} label="proc mcap" />
                    {c.raw_mcap_url && (
                      <Btn
                        title={"Copy URL: " + serverUrl + c.raw_mcap_url}
                        onClick={() => copyMcapUrl(c.raw_mcap_url!)}
                      >
                        Copy raw MCAP URL
                      </Btn>
                    )}
                    {c.processed_mcap_url && (
                      <Btn
                        title={"Copy URL: " + serverUrl + c.processed_mcap_url}
                        onClick={() => copyMcapUrl(c.processed_mcap_url!)}
                      >
                        Copy processed MCAP URL
                      </Btn>
                    )}
                    {c.has_scenario && (
                      <Btn
                        primary
                        title="Launch ESMini with this chunk's scenario"
                        onClick={() =>
                          postAction(
                            `/api/sessions/${s.id}/esmini`,
                            "ESMini",
                            { chunk: c.name },
                          )
                        }
                      >
                        ▶ ESMini
                      </Btn>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Jobs */}
        {jobs.length > 0 && (
          <>
            <div style={S.sectionTitle}>Jobs</div>
            {jobs.map((j) => (
              <React.Fragment key={j.job_id}>
                <div style={S.jobRow}>
                  <span style={{ flex: 1, color: "#ccc" }}>{j.label}</span>
                  <span style={{ color: statusColor[j.status] }}>{j.status}</span>
                  <span style={{ color: "#555" }}>{j.elapsed}s</span>
                  <Btn onClick={() => toggleLog(j.job_id)}>log</Btn>
                </div>
                {openLogs[j.job_id] && (
                  <div style={S.logBox}>
                    {openLogs[j.job_id].join("\n") || "(no output yet)"}
                  </div>
                )}
              </React.Fragment>
            ))}
          </>
        )}
      </div>

      {/* Toast */}
      {toast && <div style={S.toast}>{toast}</div>}
    </div>
  );
}

// ── Panel init ────────────────────────────────────────────────────────────────

export function initPanel(context: PanelExtensionContext): () => void {
  const state = context.initialState as PanelState | undefined;
  const initialUrl = state?.serverUrl ?? "http://localhost:8765";

  const root = createRoot(context.panelElement);

  function App() {
    const [serverUrl, setServerUrl] = useState(initialUrl);

    const handleUrlChange = useCallback((url: string) => {
      setServerUrl(url);
      context.saveState({ serverUrl: url } satisfies PanelState);
    }, []);

    return (
      <SessionPanel serverUrl={serverUrl} onUrlChange={handleUrlChange} />
    );
  }

  root.render(<App />);

  return () => {
    root.unmount();
  };
}
