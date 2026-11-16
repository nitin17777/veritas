import { useState } from "react";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";

export function ResearchPage() {
  const [selectedMethod, setSelectedMethod] = useState<number | null>(null);

  const methods = [
    {
      num: "01",
      title: "BM25 keyword retrieval",
      desc: "Finds documents with strong lexical overlap, especially names, dates, numbers, and exact phrases.",
      details: "Uses rank-bm25 with tuned parameters k1=1.5 and b=0.75, operating over stemmed token sequences. Crucial for pinpointing entity names and specific quotes.",
    },
    {
      num: "02",
      title: "Dense semantic retrieval",
      desc: "Finds semantically similar evidence even when sources use completely different vocabulary or paraphrase.",
      details: "Utilizes BAAI/bge-small-en-v1.5 embeddings with FAISS CPU/GPU cosine similarity index, mapping statements into 384-dimensional dense semantic vectors.",
    },
    {
      num: "03",
      title: "Reciprocal Rank Fusion",
      desc: "Combines keyword and semantic result lists mathematically without letting one method dominate the candidate pool.",
      details: "Applies standard RRF formula RRF(d) = sum(1 / (60 + rank_i(d))), harmonizing disparate scoring distributions without score calibration issues.",
    },
    {
      num: "04",
      title: "Source credibility analysis",
      desc: "Evaluates provenance, editorial transparency, primary fact-checking signatories, and institutional standards.",
      details: "Tier-based trust weighting from configs/credibility_tiers.yaml. Assigns provisional scores from Primary Sources (snopes, who.int, cdc.gov) to mainstream media.",
    },
    {
      num: "05",
      title: "Temporal validity scoring",
      desc: "Checks whether evidence was published in applicable temporal proximity to the claim's time window.",
      details: "Models temporal recency via exponential decay: score = exp(-|claim_date - evidence_date| / tau), with configurable half-lives from breaking news (14d) to science (730d).",
    },
    {
      num: "06",
      title: "Faithfulness & conservative abstention",
      desc: "Tests explanations against cited evidence and returns Unverified whenever support is insufficient.",
      details: "Sentence-level NLI entailment gate. If model explanations introduce hallucinations not entailed by cited evidence, the system abstains to Unverified.",
    },
  ];

  const pipeline = [
    { name: "Claim input", detail: "Natural language statement" },
    { name: "Claim extraction", detail: "Check-worthiness filter" },
    { name: "Hybrid retrieval", detail: "BM25 + Dense FAISS" },
    { name: "RRF fusion", detail: "Reciprocal rank merge" },
    { name: "Cross-encoder", detail: "Fine-grained reranking" },
    { name: "Evidence scoring", detail: "Relevance + Cred. + Time" },
    { name: "Constrained LLM", detail: "Evidence-only generation" },
    { name: "NLI validation", detail: "Faithfulness verification" },
    { name: "Five-class verdict", detail: "Inspectable provenance" },
  ];

  return (
    <div className="page research-page">
      <PageHeader
        title="Research & methodology"
        description="How VERITAS turns claims into evidence-grounded, inspectable verdicts."
      />

      <section className="research-hero">
        <span className="eyebrow">EVIDENCE, BEFORE ANSWERS</span>
        <h2>A verification system designed to make uncertainty visible.</h2>
        <p>
          VERITAS combines information retrieval, source evaluation, and constrained language
          models. It is built to show its work—not ask for blind trust.
        </p>
      </section>

      <section className="architecture card">
        <div className="section-heading">
          <div>
            <h2>Verification architecture</h2>
            <p>A layered pipeline designed to reduce unsupported conclusions.</p>
          </div>
          <span className="version-tag">VERITAS PIPELINE v1.4</span>
        </div>

        <div className="architecture-flow">
          {pipeline.map((p, i) => (
            <div className="arch-item" key={p.name}>
              <div className={`arch-node ${i === pipeline.length - 1 ? "final" : ""}`}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                <strong>{p.name}</strong>
                <small>{p.detail}</small>
              </div>
              {i < pipeline.length - 1 && (
                <span className="flow-line">
                  <Icon name="arrow" size={14} />
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="methodology">
        <div className="section-heading">
          <div>
            <h2>Methods, without the jargon</h2>
            <p>Technical rigor explained in practical terms.</p>
          </div>
        </div>

        <div className="method-grid">
          {methods.map((m, idx) => (
            <article className="method-card card" key={m.title}>
              <span>{m.num}</span>
              <h3>{m.title}</h3>
              <p>{m.desc}</p>
              {selectedMethod === idx && (
                <div style={{ marginTop: "0.8rem", padding: "0.8rem", background: "var(--bg)", borderRadius: "8px", fontSize: "0.86rem", color: "var(--muted)", lineHeight: 1.5 }}>
                  {m.details}
                </div>
              )}
              <button
                type="button"
                onClick={() => setSelectedMethod(selectedMethod === idx ? null : idx)}
              >
                {selectedMethod === idx ? "Show less" : "Learn more"} <Icon name="arrow" size={14} />
              </button>
            </article>
          ))}
        </div>
      </section>

      <section className="principle card">
        <span className="principle-mark">“</span>
        <div>
          <span className="eyebrow">CORE PRINCIPLE</span>
          <h2>
            VERITAS should not merely tell people what to believe. It should show the evidence,
            explain the reasoning, and make uncertainty visible.
          </h2>
        </div>
      </section>
    </div>
  );
}
