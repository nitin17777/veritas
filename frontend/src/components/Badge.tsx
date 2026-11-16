import type { Verdict } from "../types";

export function Badge({ verdict }: { verdict: Verdict }) {
  const norm = verdict.toLowerCase().replace(" ", "-");
  return (
    <span className={`badge badge-${norm}`}>
      <span className="badge-dot" />
      {verdict}
    </span>
  );
}
