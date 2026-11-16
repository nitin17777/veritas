import { useMemo, useState } from "react";
import type { HistoryItem, VerificationResult, Verdict } from "../types";
import { Badge } from "../components/Badge";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";

interface HistoryPageProps {
  history: HistoryItem[];
  onResult: (result: VerificationResult) => void;
  onNew: () => void;
  onClear: () => void;
}

export function HistoryPage({ history, onResult, onNew, onClear }: HistoryPageProps) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All verdicts");
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");

  const filtered = useMemo(() => {
    let items = history.filter(
      (x) =>
        x.claim.toLowerCase().includes(query.toLowerCase()) &&
        (filter === "All verdicts" || x.verdict === filter)
    );
    if (sortOrder === "oldest") {
      items = [...items].reverse();
    }
    return items;
  }, [history, query, filter, sortOrder]);

  return (
    <div className="page">
      <PageHeader
        title="Verification history"
        description="Search, revisit, and compare your previous evidence-backed fact checks."
      >
        <button className="btn btn-secondary" onClick={onClear} type="button" disabled={history.length === 0}>
          Clear history
        </button>
        <button className="btn btn-primary" onClick={onNew} type="button">
          New verification <Icon name="arrow" size={16} />
        </button>
      </PageHeader>

      <div className="toolbar card">
        <label className="search-field">
          <Icon name="search" size={17} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by claim text..."
          />
        </label>

        <label className="select-wrap">
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option>All verdicts</option>
            {(["TRUE", "PARTIALLY TRUE", "MISLEADING", "FALSE", "UNVERIFIED"] as Verdict[]).map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
          <Icon name="chevron" size={15} />
        </label>

        <label className="select-wrap sort-select">
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as "newest" | "oldest")}
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
          <Icon name="chevron" size={15} />
        </label>
      </div>

      <div className="table-card card">
        <div className="table-head">
          <span>Claim</span>
          <span>Verdict</span>
          <span>Confidence</span>
          <span>Date</span>
          <span>Evidence</span>
          <span />
        </div>

        {filtered.length > 0 ? (
          filtered.map((item, i) => (
            <div className="table-row" key={item.id || item.claim}>
              <div>
                <strong>{item.claim}</strong>
                <span>Verification #{1048 - i}</span>
              </div>
              <Badge verdict={item.verdict} />
              <div className="table-confidence">
                <span>
                  <i style={{ width: `${item.confidence}%` }} />
                </span>
                <strong>{item.confidence}%</strong>
              </div>
              <span>{item.date}</span>
              <span>{item.evidence} source{item.evidence === 1 ? "" : "s"}</span>
              <button
                aria-label="View verification"
                onClick={() => onResult(item.result)}
                type="button"
              >
                <Icon name="arrow" size={16} />
              </button>
            </div>
          ))
        ) : (
          <div className="empty-state">
            <span>
              <Icon name="search" size={26} />
            </span>
            <h3>No verifications found</h3>
            <p>
              {history.length === 0
                ? "No verifications have been run yet. Submit a claim to get started!"
                : "Try changing your search query or verdict filter."}
            </p>
          </div>
        )}
      </div>

      <p className="table-caption">
        Showing {filtered.length} of {history.length} verification{history.length === 1 ? "" : "s"}
      </p>
    </div>
  );
}
