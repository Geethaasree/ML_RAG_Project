import React, { useState } from "react";

export default function App() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState(null);
  async function ask() {
    const r = await fetch("/api/rag/query", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: q, top_k: 3 })
    });
    setRes(await r.json());
  }
  return (
    <div style={{padding:20}}>
      <h2>RAG Demo</h2>
      <textarea rows={6} cols={80} value={q} onChange={(e)=>setQ(e.target.value)} />
      <br/>
      <button onClick={ask}>Ask</button>
      <pre>{JSON.stringify(res, null, 2)}</pre>
    </div>
  );
}
