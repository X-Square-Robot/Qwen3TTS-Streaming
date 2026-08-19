import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("json", json);
hljs.registerLanguage("python", python);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("yaml", yaml);

const LANGUAGE_ALIASES = {
  bash: "bash", console: "bash", sh: "bash", shell: "bash", zsh: "bash",
  js: "javascript", javascript: "javascript", jsx: "javascript",
  json: "json", jsonc: "json",
  py: "python", python: "python",
  ts: "typescript", tsx: "typescript", typescript: "typescript",
  yaml: "yaml", yml: "yaml",
};

const LANGUAGE_LABELS = {
  bash: "Shell", javascript: "JavaScript", json: "JSON", python: "Python",
  typescript: "TypeScript", yaml: "YAML",
};

export function configureCodeRendering(markdown, documentSlug) {
  let codeBlockIndex = 0;
  markdown.renderer.rules.fence = (tokens, index) => renderCodeFence(
    tokens[index],
    `${documentSlug}-${codeBlockIndex++}`,
    markdown,
  );
  return markdown;
}

function renderCodeFence(token, id, markdown) {
  const requested = String(token.info ?? "").trim().split(/\s+/, 1)[0].toLowerCase();
  const language = LANGUAGE_ALIASES[requested];
  const highlighted = language
    ? hljs.highlight(token.content, {language, ignoreIllegals: true}).value
    : markdown.utils.escapeHtml(token.content);
  const label = LANGUAGE_LABELS[language] ?? (requested ? requested.toUpperCase() : "TEXT");
  const safeId = markdown.utils.escapeHtml(id);
  return `<div class="code-frame" data-code-id="${safeId}">`
    + `<div class="code-toolbar"><span>${markdown.utils.escapeHtml(label)}</span>`
    + `<button type="button" data-copy-code>COPY</button></div>`
    + `<pre class="hljs"><code class="language-${language ?? "text"}">${highlighted}</code></pre>`
    + `</div>\n`;
}
