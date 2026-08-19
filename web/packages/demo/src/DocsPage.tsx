import {useEffect, useMemo, useState, type MouseEvent} from "react";

type DocLocale = "zh-CN" | "en";
type DocLayer = "start" | "advanced" | "details";

interface BuiltDocument {
  readonly slug: string;
  readonly topic: string;
  readonly locale: DocLocale;
  readonly layer: DocLayer;
  readonly title: string;
  readonly summary: string;
  readonly source: string;
  readonly html: string;
}

interface BuiltDocs {
  readonly schema_version: "qwen.tts.docs.v1";
  readonly documents: BuiltDocument[];
}

const LAYERS: Array<{id: DocLayer; index: string; zh: string; en: string; zhHint: string; enHint: string}> = [
  {id: "start", index: "01", zh: "快速接入", en: "Get started", zhHint: "先跑通一次", enHint: "Complete one request"},
  {id: "advanced", index: "02", zh: "高级配置", en: "Configure", zhHint: "再贴合业务", enHint: "Tune for your product"},
  {id: "details", index: "03", zh: "更多细节", en: "Reference", zhHint: "需要时再查", enHint: "Look up when needed"},
];

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

  const requestedLocale: DocLocale = slug.endsWith("-en") ? "en" : "zh-CN";
  const selectedDocument = docs?.documents.find((entry) => entry.slug === slug)
    ?? docs?.documents.find((entry) => entry.locale === requestedLocale)
    ?? docs?.documents[0];
  const locale = selectedDocument?.locale ?? requestedLocale;
  const localizedDocuments = useMemo(
    () => docs?.documents.filter((entry) => entry.locale === locale) ?? [],
    [docs, locale],
  );
  const isChinese = locale === "zh-CN";

  useEffect(() => {
    const query = new URLSearchParams(window.location.hash.split("?", 2)[1] ?? "");
    const anchor = query.get("anchor");
    if (selectedDocument && anchor) requestAnimationFrame(() => window.document.getElementById(anchor)?.scrollIntoView());
  }, [selectedDocument]);

  function switchLocale(nextLocale: DocLocale) {
    if (!docs || nextLocale === locale) return;
    const counterpart = docs.documents.find((entry) => (
      entry.locale === nextLocale && entry.topic === selectedDocument?.topic
    )) ?? docs.documents.find((entry) => entry.locale === nextLocale);
    if (counterpart) window.location.hash = `#/docs/${counterpart.slug}`;
  }

  async function copyCode(event: MouseEvent<HTMLElement>) {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-copy-code]");
    if (!button) return;
    const code = button.closest(".code-frame")?.querySelector("code")?.textContent;
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      button.textContent = isChinese ? "已复制" : "COPIED";
    } catch {
      button.textContent = isChinese ? "复制失败" : "FAILED";
    }
    window.setTimeout(() => { button.textContent = "COPY"; }, 1300);
  }

  return <section className="page docs-page">
    <header className="docs-hero">
      <div className="docs-hero-copy">
        <p className="eyebrow"><span/>{isChinese ? "调用方文档" : "INTEGRATION GUIDE"}</p>
        <h1>{isChinese ? "先让声音出来，再按需深入。" : "Make it speak. Then go deeper."}</h1>
        <p>{isChinese
          ? "默认路径只讲完成调用所需的内容。参数、协议和限制各自在下一层等你。"
          : "The default path contains only what you need for a working request. Configuration and protocol details wait in the next layers."}</p>
      </div>
      <div className="docs-language" role="group" aria-label={isChinese ? "文档语言" : "Documentation language"}>
        <button className={isChinese ? "active" : ""} onClick={() => switchLocale("zh-CN")}>中文</button>
        <button className={!isChinese ? "active" : ""} onClick={() => switchLocale("en")}>EN</button>
      </div>
      <div className="integration-track" aria-label={isChinese ? "接入路径" : "Integration path"}>
        {(isChinese ? ["连接服务", "发送文本", "收到音频"] : ["CONNECT", "SEND TEXT", "RECEIVE AUDIO"]).map((label, index) => <div className="track-step" key={label}>
          <span>{index + 1}</span><strong>{label}</strong>{index < 2 && <i/>}
        </div>)}
      </div>
    </header>

    {error && <p className="alert">{error}</p>}
    <div className="docs-layout">
      <label className="docs-mobile-nav">
        <span>{isChinese ? "选择文档章节" : "Choose a section"}</span>
        <select value={selectedDocument?.slug ?? ""} onChange={(event) => {
          window.location.hash = `#/docs/${event.target.value}`;
        }}>
          {LAYERS.map((layer) => {
            const entries = localizedDocuments.filter((entry) => entry.layer === layer.id);
            return entries.length > 0 && <optgroup label={`${layer.index} ${isChinese ? layer.zh : layer.en}`} key={layer.id}>
              {entries.map((entry) => <option value={entry.slug} key={entry.slug}>{entry.title}</option>)}
            </optgroup>;
          })}
        </select>
      </label>
      <aside className="docs-nav" aria-label={isChinese ? "文档层级" : "Documentation levels"}>
        {LAYERS.map((layer) => {
          const entries = localizedDocuments.filter((entry) => entry.layer === layer.id);
          if (entries.length === 0) return null;
          return <section className="docs-nav-group" key={layer.id}>
            <header><span>{layer.index}</span><div><strong>{isChinese ? layer.zh : layer.en}</strong><small>{isChinese ? layer.zhHint : layer.enHint}</small></div></header>
            {entries.map((entry) => <a
              className={entry.slug === selectedDocument?.slug ? "active" : ""}
              href={`#/docs/${entry.slug}`}
              key={entry.slug}
            ><strong>{entry.title}</strong><small>{entry.summary}</small></a>)}
          </section>;
        })}
      </aside>
      <article className="panel markdown docs-article" onClick={(event) => void copyCode(event)}>
        {selectedDocument && <div className="document-meta">
          <span>{LAYERS.find((entry) => entry.id === selectedDocument.layer)?.[isChinese ? "zh" : "en"]}</span>
          <span>{isChinese ? "随服务版本发布" : "SHIPS WITH THIS SERVICE"}</span>
        </div>}
        {selectedDocument
          ? <div dangerouslySetInnerHTML={{__html: selectedDocument.html}} />
          : <div className="docs-loading"><span/><p>{isChinese ? "正在加载接入文档…" : "Loading integration guide…"}</p></div>}
      </article>
    </div>
  </section>;
}

function docSlugFromHash(): string {
  const slug = window.location.hash.match(/^#\/docs\/([^?]+)/)?.[1];
  if (slug === "overview-zh") return "quickstart-zh";
  if (slug === "overview-en") return "quickstart-en";
  return slug ?? "quickstart-zh";
}
