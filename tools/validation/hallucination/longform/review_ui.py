"""Offline, arm-blind single-sample review interface generation."""

from __future__ import annotations

import hashlib
import html
import json
from typing import Any, Mapping, Sequence

from .models import ReviewLabel


_LABEL_HELP = {
    ReviewLabel.OK: "未发现列出的听感问题",
    ReviewLabel.SINGLE_UNIT_LOOP: "参考外单音节连续重复或持续循环",
    ReviewLabel.ABNORMAL_NOISE: "持续的异常非语言噪音",
    ReviewLabel.UNSUPPORTED_SPEECH: "确认存在参考外语音插入",
    ReviewLabel.OMISSION: "仅有内容遗漏",
    ReviewLabel.MISPRONUNCIATION: "仅有误读或发音问题",
    ReviewLabel.UNSCORABLE: "当前音频确实无法裁决",
}


def build_review_page(
    rows: Sequence[Mapping[str, Any]],
    *,
    public_fields: Sequence[str],
) -> str:
    public_rows = [
        {field: str(row.get(field, "") or "") for field in public_fields}
        for row in rows
    ]
    data_json = json.dumps(public_rows, ensure_ascii=False, separators=(",", ":"))
    # A script-data element still recognizes a literal closing script tag.  Escaping
    # HTML-significant characters keeps reference text inert without changing JSON.
    data_json = (
        data_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    labels_json = json.dumps(
        [item.value for item in ReviewLabel], ensure_ascii=True, separators=(",", ":")
    )
    package_id = hashlib.sha256(data_json.encode("utf-8")).hexdigest()[:16]
    label_controls = "".join(
        (
            f'<label class="decision-option" data-value="{item.value}">'
            f'<input type="radio" name="review-label" value="{item.value}">'
            f'<span class="shortcut" aria-hidden="true">{index}</span>'
            f'<span><strong>{item.value}</strong>'
            f'<small>{html.escape(_LABEL_HELP[item])}</small></span></label>'
        )
        for index, item in enumerate(ReviewLabel, start=1)
    )
    return (
        r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>0818 长文本盲审 · 听辨工作台</title>
<style>
:root{color-scheme:light;--bench:#e7edf0;--paper:#f9fbfc;--ink:#18262d;--muted:#617178;--rule:#b8c7cd;--signal:#087f8c;--signal-soft:#d7ecee;--warn:#a4432e}
*{box-sizing:border-box}html{background:var(--bench)}body{margin:0;color:var(--ink);background:linear-gradient(90deg,transparent 0 49.9%,rgba(24,38,45,.035) 50%,transparent 50.1%);font:16px/1.55 "Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif}
button,input,textarea{font:inherit;color:inherit}button,.button{border:1px solid var(--ink);border-radius:3px;background:var(--paper);padding:.64rem .9rem;font-weight:650;cursor:pointer}.button{display:inline-flex;align-items:center;text-decoration:none}button:hover,.button:hover{background:var(--signal-soft)}button:disabled{cursor:not-allowed;opacity:.42;background:transparent}
:focus-visible{outline:3px solid var(--signal);outline-offset:3px}.shell{width:min(1180px,calc(100% - 2rem));margin:0 auto;padding:1.25rem 0 2.5rem}.masthead{display:flex;justify-content:space-between;gap:2rem;align-items:end;padding:.5rem 0 1.15rem;border-bottom:1px solid var(--rule)}
.eyebrow,.utility,.sample-code,.position,.shortcut,kbd{font-family:ui-monospace,"Cascadia Mono",monospace}.eyebrow{margin:0 0 .25rem;color:var(--signal);font-size:.72rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase}.masthead h1{margin:0;font-size:clamp(1.55rem,4vw,2.5rem);font-weight:760;letter-spacing:-.035em}.masthead p{max-width:34rem;margin:0;color:var(--muted)}
.transmission{margin:1.15rem 0 1.4rem;padding:.8rem 1rem;background:var(--ink);color:var(--paper);border-radius:3px}.transmission-copy{display:flex;justify-content:space-between;gap:1rem;align-items:baseline;margin-bottom:.55rem}.transmission-copy strong{font-size:.82rem;letter-spacing:.08em}.transmission-copy output{font:700 1rem ui-monospace,monospace}.transmission-track{position:relative;height:18px;overflow:hidden;background:#43525a;border:1px solid #708087}.transmission-fill{height:100%;width:0;background:var(--signal);transition:width .22s ease}.transmission-track::after{position:absolute;inset:0;content:"";background:repeating-linear-gradient(90deg,transparent 0 13px,rgba(249,251,252,.22) 13px 14px);pointer-events:none}
.toolbar{display:flex;flex-wrap:wrap;gap:.55rem;align-items:center;margin-bottom:1rem}.toolbar .spacer{flex:1}.save-status{min-width:15rem;color:var(--muted);font-size:.85rem;text-align:right}.visually-hidden{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important}
.workbench{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(19rem,.65fr);gap:1rem;align-items:start}.sample-panel,.decision-panel{background:var(--paper);border:1px solid var(--rule);border-radius:4px;box-shadow:0 10px 28px rgba(24,38,45,.06)}.sample-panel{padding:clamp(1.15rem,3vw,2rem)}.sample-meta{display:flex;justify-content:space-between;gap:1rem;color:var(--muted);font-size:.8rem}.sample-code{color:var(--signal);font-weight:750}.reference-label{margin:2rem 0 .3rem;color:var(--muted);font-size:.76rem;font-weight:750;letter-spacing:.08em}.reference{margin:.2rem 0 1.75rem;font-size:clamp(1.3rem,3vw,2rem);font-weight:650;line-height:1.55;letter-spacing:.015em}.audio-block{padding:1rem;border-left:4px solid var(--signal);background:#eef4f5}.audio-block label{display:block;margin-bottom:.55rem;font-weight:720}.audio-block audio{display:block;width:100%}
details{margin-top:1.2rem;border-top:1px solid var(--rule);padding-top:1rem}summary{cursor:pointer;font-weight:700}.context-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.9rem}.context-grid section{padding:.75rem;background:var(--bench)}.context-grid h3{margin:0 0 .35rem;color:var(--muted);font-size:.74rem;letter-spacing:.08em}.context-grid p{margin:0}.context-audio{margin-top:.8rem;width:100%}
.decision-panel{padding:1rem;position:sticky;top:1rem}.decision-panel fieldset{padding:0;margin:0;border:0}.decision-panel legend{padding:0 0 .75rem;font-size:1rem;font-weight:800}.decision-option{display:grid;grid-template-columns:1.7rem 1fr;gap:.65rem;align-items:start;margin:.35rem 0;padding:.62rem;border:1px solid var(--rule);border-radius:3px;cursor:pointer}.decision-option:has(input:checked){border-color:var(--signal);background:var(--signal-soft)}.decision-option input{position:absolute;opacity:0;pointer-events:none}.decision-option:has(input:focus-visible){outline:3px solid var(--signal);outline-offset:2px}.decision-option strong{display:block;font:700 .75rem ui-monospace,"Cascadia Mono",monospace;overflow-wrap:anywhere}.decision-option small{display:block;margin-top:.08rem;color:var(--muted);font-size:.75rem}.shortcut{display:grid;place-items:center;width:1.55rem;height:1.55rem;border:1px solid var(--rule);background:var(--paper);font-size:.72rem;font-weight:800}
.notes-label{display:block;margin-top:1rem;font-weight:750}.notes-label span{display:block;margin-bottom:.35rem}.notes{width:100%;min-height:5.3rem;resize:vertical;border:1px solid var(--rule);border-radius:3px;background:white;padding:.65rem}.navigation{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:1rem}.navigation .next-incomplete{grid-column:1/-1;border-color:var(--signal);color:var(--signal)}.shortcuts{margin:1rem 0 0;color:var(--muted);font-size:.75rem}.shortcuts kbd{display:inline-block;min-width:1.5rem;padding:.08rem .3rem;border:1px solid var(--rule);border-bottom-width:2px;border-radius:2px;background:white;color:var(--ink);text-align:center}
.message{min-height:1.5rem;margin:.75rem 0 0;color:var(--muted);font-size:.82rem}.message[data-tone="error"]{color:var(--warn);font-weight:700}
@media(max-width:800px){.masthead{display:block}.masthead p{margin-top:.5rem}.workbench{grid-template-columns:1fr}.decision-panel{position:static}.save-status{order:2;width:100%;text-align:left}.context-grid{grid-template-columns:1fr}}
@media(max-width:520px){.shell{width:min(100% - 1rem,1180px)}.sample-panel{padding:1rem}.toolbar button,.toolbar .button{flex:1;justify-content:center}.transmission-copy{align-items:start}.reference{font-size:1.25rem}.decision-option{padding:.55rem}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}}
</style>
</head>
<body>
<div class="shell">
<header class="masthead"><div><p class="eyebrow">Acoustic evidence desk · 0818</p><h1>长文本听辨工作台</h1></div><p>一次只裁决一个样本。页面仅保存到当前浏览器，不会上传任何内容。</p></header>
<section class="transmission" aria-labelledby="transmission-title">
  <div class="transmission-copy"><strong id="transmission-title">审阅传输条</strong><output id="progress-copy">0 / 0 已裁决</output></div>
  <div id="progress-track" class="transmission-track" role="progressbar" aria-label="盲审完成进度" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"><div id="progress-fill" class="transmission-fill"></div></div>
</section>
<nav class="toolbar" aria-label="审阅文件操作">
  <button id="import-trigger" type="button">导入已填 CSV</button><input class="visually-hidden" id="import-file" type="file" accept=".csv,text/csv" tabindex="-1">
  <button id="export-progress" type="button">导出当前进度</button>
  <button id="export-final" type="button" disabled>导出完整终稿</button>
  <span class="spacer"></span><span id="save-status" class="save-status" role="status" aria-live="polite">正在读取本地进度…</span>
</nav>
<main class="workbench">
  <article class="sample-panel" aria-labelledby="sample-heading">
    <div class="sample-meta"><span id="sample-id" class="sample-code"></span><span id="position" class="position"></span></div>
    <p class="reference-label">本条参考句</p><h2 id="sample-heading" class="reference" tabindex="-1"></h2>
    <section class="audio-block"><label for="main-audio">待裁决音频 · <span class="utility">Space</span></label><audio id="main-audio" controls preload="metadata"></audio></section>
    <details id="context-details"><summary>需要时查看前后文回退片段</summary><div class="context-grid"><section><h3>前一句</h3><p id="previous-reference"></p></section><section><h3>后一句</h3><p id="next-reference"></p></section></div><audio id="context-audio" class="context-audio" controls preload="none"></audio></details>
  </article>
  <aside class="decision-panel" aria-label="当前样本裁决">
    <fieldset><legend>选择一个标签 <span class="utility">1–7</span></legend>"""
        + label_controls
        + r"""</fieldset>
    <label class="notes-label" for="notes"><span>备注 / 问题时间点</span><textarea id="notes" class="notes" placeholder="例如：00:04.8 开始循环"></textarea></label>
    <div class="navigation"><button id="previous" type="button">← 上一条</button><button id="next" type="button">下一条 →</button><button id="next-incomplete" class="next-incomplete" type="button">前往下一条未完成</button></div>
    <p class="shortcuts"><kbd>←</kbd><kbd>→</kbd> 导航　<kbd>1</kbd>–<kbd>7</kbd> 标签　<kbd>Space</kbd> 主音频　<kbd>Shift</kbd>+<kbd>Space</kbd> 前后文</p>
    <p id="message" class="message" role="status" aria-live="polite"></p>
  </aside>
</main>
</div>
<script id="review-data" type="application/json">"""
        + data_json
        + r"""</script>
<script>
(() => {
  'use strict';
  const samples = JSON.parse(document.getElementById('review-data').textContent);
  const allowedLabels = """
        + labels_json
        + r""";
  const allowedSet = new Set(allowedLabels);
  const storageKey = 'qwen3tts-blind-review-v1:"""
        + package_id
        + r"""';
  const byId = new Map(samples.map((sample, index) => [sample.blind_id, index]));
  const elements = Object.fromEntries(['sample-id','position','sample-heading','main-audio','context-audio','context-details','previous-reference','next-reference','notes','previous','next','next-incomplete','progress-copy','progress-track','progress-fill','save-status','message','import-trigger','import-file','export-progress','export-final'].map(id => [id, document.getElementById(id)]));
  const radios = [...document.querySelectorAll('input[name="review-label"]')];
  let state = {version: 1, current: 0, answers: {}};

  const editableTarget = target => target instanceof Element && target.closest('button,input,textarea,select,a,audio,summary,[contenteditable="true"]') !== null;
  const answerFor = id => state.answers[id] || {label: '', notes: ''};
  const completedCount = () => samples.reduce((count, sample) => count + (allowedSet.has(answerFor(sample.blind_id).label) ? 1 : 0), 0);
  function announce(text, tone = '') { elements.message.textContent = text; elements.message.dataset.tone = tone; }
  function loadLocal() {
    try {
      const restored = JSON.parse(localStorage.getItem(storageKey) || 'null');
      if (!restored || restored.version !== 1 || typeof restored.answers !== 'object') return;
      for (const [blindId, answer] of Object.entries(restored.answers)) {
        if (!byId.has(blindId) || !answer || typeof answer !== 'object') continue;
        const label = allowedSet.has(answer.label) ? answer.label : '';
        state.answers[blindId] = {label, notes: String(answer.notes || '')};
      }
      if (Number.isInteger(restored.current)) state.current = Math.max(0, Math.min(samples.length - 1, restored.current));
    } catch (error) { announce('无法读取浏览器本地进度，可导入之前导出的 CSV。', 'error'); }
  }
  function saveLocal() {
    try { localStorage.setItem(storageKey, JSON.stringify(state)); elements['save-status'].textContent = '已自动保存到当前浏览器'; }
    catch (error) { elements['save-status'].textContent = '本地保存不可用，请及时导出进度'; elements['save-status'].dataset.tone = 'error'; }
  }
  function stopOtherAudio(active) {
    [elements['main-audio'], elements['context-audio']].forEach(audio => { if (audio !== active) audio.pause(); });
  }
  function toggleAudio(audio) {
    stopOtherAudio(audio);
    if (audio.paused) audio.play().catch(() => announce('浏览器阻止了播放，请直接使用音频控件。', 'error')); else audio.pause();
  }
  function updateProgress() {
    const complete = completedCount(), total = samples.length;
    elements['progress-copy'].textContent = `${complete} / ${total} 已裁决`;
    elements['progress-track'].setAttribute('aria-valuemax', String(total));
    elements['progress-track'].setAttribute('aria-valuenow', String(complete));
    elements['progress-fill'].style.width = total ? `${complete / total * 100}%` : '0%';
    elements['export-final'].disabled = complete !== total || total === 0;
  }
  function render(focusHeading = false) {
    if (!samples.length) { announce('审阅清单为空。', 'error'); return; }
    const sample = samples[state.current], answer = answerFor(sample.blind_id);
    stopOtherAudio(null);
    elements['sample-id'].textContent = sample.blind_id;
    elements.position.textContent = `${state.current + 1} / ${samples.length}`;
    elements['sample-heading'].textContent = sample.reference_text;
    elements['previous-reference'].textContent = sample.previous_reference || '（无）';
    elements['next-reference'].textContent = sample.next_reference || '（无）';
    elements['main-audio'].src = sample.audio;
    elements['context-audio'].src = sample.context_audio;
    elements['context-details'].open = false;
    elements.notes.value = answer.notes;
    radios.forEach(radio => { radio.checked = radio.value === answer.label; });
    elements.previous.disabled = state.current === 0;
    elements.next.disabled = state.current === samples.length - 1;
    updateProgress();
    saveLocal();
    if (focusHeading) elements['sample-heading'].focus({preventScroll: true});
  }
  function moveTo(index) { state.current = Math.max(0, Math.min(samples.length - 1, index)); render(true); }
  function selectLabel(label) {
    if (!allowedSet.has(label)) return;
    const sample = samples[state.current], answer = answerFor(sample.blind_id);
    state.answers[sample.blind_id] = {label, notes: answer.notes};
    radios.forEach(radio => { radio.checked = radio.value === label; });
    updateProgress(); saveLocal(); announce(`已标记 ${label}`);
  }
  function nextIncomplete() {
    for (let step = 1; step <= samples.length; step += 1) {
      const index = (state.current + step) % samples.length;
      if (!allowedSet.has(answerFor(samples[index].blind_id).label)) { moveTo(index); return; }
    }
    announce('全部样本已完成，可以导出完整终稿。');
  }
  function quoteCsv(value) { return `"${String(value ?? '').replaceAll('"', '""')}"`; }
  function exportCsv(filename) {
    const records = [['blind_id','label','notes'], ...samples.map(sample => { const answer = answerFor(sample.blind_id); return [sample.blind_id, answer.label, answer.notes]; })];
    const csvText = records.map(record => record.map(quoteCsv).join(',')).join('\r\n');
    const blob = new Blob(['\ufeff', csvText], {type: 'text/csv;charset=utf-8'}), url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  function parseCsv(text) {
    const records = []; let record = [], field = '', quoted = false;
    const source = text.replace(/^\ufeff/, '');
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (quoted) {
        if (character === '"' && source[index + 1] === '"') { field += '"'; index += 1; }
        else if (character === '"') quoted = false;
        else field += character;
      } else if (character === '"' && field === '') quoted = true;
      else if (character === ',') { record.push(field); field = ''; }
      else if (character === '\n') { record.push(field.replace(/\r$/, '')); records.push(record); record = []; field = ''; }
      else field += character;
    }
    if (quoted) throw new Error('CSV 存在未闭合的引号');
    if (field !== '' || record.length) { record.push(field.replace(/\r$/, '')); records.push(record); }
    return records;
  }
  async function importCsv(file) {
    const records = parseCsv(await file.text());
    if (!records.length) throw new Error('CSV 为空');
    const headers = records[0].map(value => value.trim()), idIndex = headers.indexOf('blind_id'), labelIndex = headers.indexOf('label'), notesIndex = headers.indexOf('notes');
    if (idIndex < 0 || labelIndex < 0 || notesIndex < 0) throw new Error('CSV 必须包含 blind_id、label、notes 列');
    const seen = new Set(); let imported = 0;
    for (const record of records.slice(1)) {
      if (record.length === 1 && record[0] === '') continue;
      const blindId = String(record[idIndex] || '').trim(), label = String(record[labelIndex] || '').trim().toUpperCase(), notes = String(record[notesIndex] || '');
      if (!byId.has(blindId)) throw new Error(`CSV 包含未知 blind_id：${blindId || '（空）'}`);
      if (seen.has(blindId)) throw new Error(`CSV 包含重复 blind_id：${blindId}`);
      if (label && !allowedSet.has(label)) throw new Error(`CSV 标签无效：${label}`);
      seen.add(blindId); state.answers[blindId] = {label, notes}; imported += 1;
    }
    const firstIncomplete = samples.findIndex(sample => !allowedSet.has(answerFor(sample.blind_id).label));
    state.current = firstIncomplete < 0 ? 0 : firstIncomplete; render(); announce(`已从 CSV 恢复 ${imported} 条记录。`);
  }
  radios.forEach(radio => radio.addEventListener('change', () => selectLabel(radio.value)));
  elements.notes.addEventListener('input', () => { const sample = samples[state.current], answer = answerFor(sample.blind_id); state.answers[sample.blind_id] = {label: answer.label, notes: elements.notes.value}; saveLocal(); });
  elements.previous.addEventListener('click', () => moveTo(state.current - 1));
  elements.next.addEventListener('click', () => moveTo(state.current + 1));
  elements['next-incomplete'].addEventListener('click', nextIncomplete);
  elements['main-audio'].addEventListener('play', () => stopOtherAudio(elements['main-audio']));
  elements['context-audio'].addEventListener('play', () => stopOtherAudio(elements['context-audio']));
  elements['import-trigger'].addEventListener('click', () => elements['import-file'].click());
  elements['export-progress'].addEventListener('click', () => exportCsv('review_round1_progress.csv'));
  elements['export-final'].addEventListener('click', () => { if (completedCount() === samples.length) exportCsv('review_round1_filled.csv'); });
  elements['import-file'].addEventListener('change', async event => { const [file] = event.target.files || []; if (!file) return; try { await importCsv(file); } catch (error) { announce(`导入失败：${error.message}`, 'error'); } finally { event.target.value = ''; } });
  document.addEventListener('keydown', event => {
    if (editableTarget(event.target)) return;
    if (/^[1-7]$/.test(event.key)) { event.preventDefault(); selectLabel(allowedLabels[Number(event.key) - 1]); }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); moveTo(state.current - 1); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); moveTo(state.current + 1); }
    else if (event.code === 'Space') { event.preventDefault(); toggleAudio(event.shiftKey ? elements['context-audio'] : elements['main-audio']); }
  });
  loadLocal(); render();
})();
</script>
</body>
</html>
"""
    )


__all__ = ["build_review_page"]
