import {useEffect, useState} from "react";

interface BuiltDocs {
  schema_version: "qwen.tts.docs.v1";
  documents: Array<{slug: string; title: string; source: string; html: string}>;
}

export function DocsPage() {
  const [docs, setDocs] = useState<BuiltDocs | null>(null);
  const [slug, setSlug] = useState(docSlugFromHash());
  const [error, setError] = useState("");
  useEffect(() => {
    fetch("./docs.json").then(async (response) => {
      if (!response.ok) throw new Error(`Documentation failed with HTTP ${response.status}`);
      return response.json() as Promise<BuiltDocs>;
    }).then(setDocs).catch((cause) => setError(String(cause)));
    const onHash = () => setSlug(docSlugFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const selectedDocument = docs?.documents.find((entry) => entry.slug === slug) ?? docs?.documents[0];
  useEffect(() => {
    const query = new URLSearchParams(window.location.hash.split("?", 2)[1] ?? "");
    const anchor = query.get("anchor");
    if (selectedDocument && anchor) requestAnimationFrame(() => window.document.getElementById(anchor)?.scrollIntoView());
  }, [selectedDocument]);
  return <section className="page docs-page"><p className="eyebrow">DOCUMENTATION</p><h1>同一份源码，同一个版本。</h1>
    {error && <p className="alert">{error}</p>}
    <div className="docs-layout"><aside>{docs?.documents.map((entry) =>
      <a className={entry.slug === selectedDocument?.slug ? "active" : ""} href={`#/docs/${entry.slug}`} key={entry.slug}>
        {entry.title}<small>{entry.source}</small>
      </a>)}</aside>
      <article className="panel markdown">{selectedDocument
        ? <div dangerouslySetInnerHTML={{__html: selectedDocument.html}} />
        : <p>正在加载同版本文档…</p>}</article>
    </div></section>;
}

function docSlugFromHash(): string {
  return window.location.hash.match(/^#\/docs\/([^?]+)/)?.[1] ?? "overview-zh";
}
