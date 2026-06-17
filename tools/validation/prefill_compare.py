#!/usr/bin/env python3
"""Unified prefill comparison tool.

Merges official-prefill, compare-live-vs-exported-prefill,
compare-prefill-paths, official-vs-manual-rollout, and cp-sampled-parity
into one entry-point selected via --mode.

Modes:
  official          Build prefill like model.generate() (L2086-2232).
  live-vs-exported  Compare live model embeddings vs exported weight files.
  compare-paths     Compare official prefill path vs engine builder path.
  manual-rollout    Compare official model.generate() vs manual step-by-step.
  cp-parity         Compare official CP generate vs unrolled CP on sampled inputs.

Deprecated: official_prefill.py, compare_live_vs_exported_prefill.py,
compare_prefill_paths.py, official_vs_manual_rollout.py, cp_sampled_parity.py.
Use ``python -m tests.tools.prefill_compare --mode <mode>`` instead.
"""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

import torch, torch.nn.functional as F

from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports
bootstrap_project_imports("repo", "scripts_python", "third_party_qwen")

# -- shared -----------------------------------------------------------------

def _print_deprecation():
    print("NOTE: official_prefill.py, compare_live_vs_exported_prefill.py, "
          "compare_prefill_paths.py, official_vs_manual_rollout.py, and "
          "cp_sampled_parity.py are deprecated. "
          "Use  python -m tests.tools.prefill_compare --mode <mode>  instead.", file=sys.stderr)

def tensor_diff(a, b) -> dict:
    af, bf = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    d = (af - bf).abs()
    return {"shape_a": list(a.shape), "shape_b": list(b.shape),
            "max": float(d.max().cpu()), "mean": float(d.mean().cpu()),
            "cosine": float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())}

def _load_model(model_dir, device="cuda:0"):
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    m = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
    m.eval(); return m

def _load_builder(model_dir, weights_dir, device):
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder
    tok = load_lightweight_tokenizer(model_dir)
    w = EmbeddingWeights(weights_dir, device_id=device.index or 0)
    return PrefillBuilder(w, tok), w, tok

# -- mode: official (build_prefill_like_official) ---------------------------

def build_prefill_like_official(model, input_id, language, speaker, device, *,
                                instruct_ids=None, non_streaming_mode=False):
    """Replicate model.generate() prefill logic (L2086-2232)."""
    talker, cfg = model.talker, model.config
    tc = getattr(cfg, "talker_config", None)
    if tc is None: raise RuntimeError("Model has no talker_config")
    input_id = input_id.to(device)
    # speaker embed
    spk_embed = None
    if speaker and speaker.lower() in (getattr(tc, "spk_id", {}) or {}):
        spk_embed = talker.get_input_embeddings()(
            torch.tensor(tc.spk_id[speaker.lower()], device=device, dtype=input_id.dtype))
    # language
    lang_id = None
    if language and language.lower() != "auto":
        lang_id = (getattr(tc, "codec_language_id", {}) or {}).get(language.lower())
    # special embeds
    bos_e, eos_e, pad_e = talker.text_projection(talker.get_text_embeddings()(
        torch.tensor([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                     device=device, dtype=input_id.dtype))).chunk(3, dim=1)
    # codec prefill
    cpl = [[tc.codec_think_id, tc.codec_think_bos_id, lang_id, tc.codec_think_eos_id]] \
        if lang_id is not None else [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]]
    c0 = talker.get_input_embeddings()(torch.tensor(cpl, device=device, dtype=input_id.dtype))
    c1 = talker.get_input_embeddings()(torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=input_id.dtype))
    codec_in = torch.cat([c0, spk_embed.view(1,1,-1), c1], dim=1) if spk_embed is not None else torch.cat([c0, c1], dim=1)
    # role + pad+bos + codec
    role_e = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    pre = torch.cat((pad_e.expand(-1, codec_in.shape[1]-2, -1), bos_e), dim=1) + codec_in[:, :-1]
    tie = torch.cat((role_e, pre), dim=1)
    # instruct
    if instruct_ids is not None:
        instruct_ids = instruct_ids.to(device=device, dtype=input_id.dtype)
        if instruct_ids.dim() == 1: instruct_ids = instruct_ids.unsqueeze(0)
        tie = torch.cat((talker.text_projection(talker.get_text_embeddings()(instruct_ids)), tie), dim=1)
    # first text token + codec_bos
    tie = torch.cat([tie, talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_in[:, -1:]], dim=1)
    # trailing
    if non_streaming_mode:
        tie = tie[:, :-1]
        tp = input_id[:, 3:-5]; nt = tp.shape[1]
        te = torch.cat((talker.text_projection(talker.get_text_embeddings()(tp)), eos_e), dim=1)
        cpe = talker.get_input_embeddings()(torch.full((1, nt+1), tc.codec_pad_id, device=device, dtype=input_id.dtype))
        tie = torch.cat([tie, te+cpe, pad_e + talker.get_input_embeddings()(
            torch.tensor([[tc.codec_bos_id]], device=device, dtype=input_id.dtype))], dim=1)
        return tie, [pad_e.clone()]
    mid = input_id[:, 4:-5] if input_id.shape[1] > 9 else input_id[:, :0]
    tth = torch.cat((talker.text_projection(talker.get_text_embeddings()(mid)), eos_e), dim=1) if mid.shape[1] > 0 else eos_e
    return tie, [tth[:, i:i+1, :].clone() for i in range(tth.shape[1])]

# -- mode: live-vs-exported -------------------------------------------------

def _run_live_vs_exported(args):
    from engine.backend.prefill import OFFICIAL_ASSISTANT_FMT, TaskType
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = torch.device("cuda:0")
    model = _load_model(model_dir); talker = model.talker
    builder, weights, tok = _load_builder(model_dir, str(REPO_ROOT / "workspace/exported/custom-1.7b/weights"), device)
    raw_sp = torch.load(REPO_ROOT / "workspace/exported/custom-1.7b/weights/special_embeddings.pt",
                        map_location=device, weights_only=True)
    ids = torch.as_tensor(tok(OFFICIAL_ASSISTANT_FMT.format(text=args.text), return_tensors="pt")["input_ids"],
                          device=device, dtype=torch.long)
    if ids.dim() == 1: ids = ids.unsqueeze(0)
    text_ids = builder._encode_text_ids(args.text)
    off_pf, off_tr = build_prefill_like_official(model, ids, "auto", "vivian", device)
    plan = builder.build_plan_from_ids(task_type=TaskType.CUSTOM_VOICE, token_ids=text_ids,
                                       language="auto", speaker="vivian", include_eos=True)
    eng_pf = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    eng_tr = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]
    # component diffs
    sp_ids = torch.tensor([[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                          device=device, dtype=torch.long)
    off_sp = talker.text_projection(talker.get_text_embeddings()(sp_ids))
    rt_sp = torch.cat([weights.tts_bos_embed, weights.tts_eos_embed, weights.tts_pad_embed], dim=1)
    ex_sp = torch.cat([raw_sp[k].to(device=device, dtype=torch.bfloat16)
                       for k in ("tts_bos_embed", "tts_eos_embed", "tts_pad_embed")], dim=1)
    tc = model.config.talker_config
    c_ids = torch.tensor([[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id,
                           tc.spk_id["vivian"], tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=torch.long)
    # forward
    suppress = [i for i in range(talker.config.vocab_size-1024, talker.config.vocab_size)
                if i not in (talker.config.codec_eos_token_id,)]
    hist = torch.empty((1,0), device=device, dtype=torch.long)
    with torch.no_grad():
        o1 = talker.model(inputs_embeds=off_pf.to(device=device, dtype=torch.bfloat16), use_cache=True, return_dict=True)
        o2 = talker.model(inputs_embeds=eng_pf, use_cache=True, return_dict=True)
    r1 = talker.codec_head(o1.last_hidden_state)[:, -1, :]; r2 = talker.codec_head(o2.last_hidden_state)[:, -1, :]
    p1 = RepetitionPenaltyLogitsProcessor(args.repetition_penalty)(hist, r1.float().clone())
    p2 = RepetitionPenaltyLogitsProcessor(args.repetition_penalty)(hist, r2.float().clone())
    s = {"component_diffs": {
            "tts_specials_exported": tensor_diff(off_sp, ex_sp),
            "tts_specials_runtime": tensor_diff(off_sp, rt_sp),
            "codec_prefill": tensor_diff(talker.get_input_embeddings()(c_ids), weights.codec_embed(c_ids))},
         "prefill_diff": tensor_diff(off_pf, eng_pf),
         "trailing_diffs": [tensor_diff(a, b) for a, b in zip(off_tr, eng_tr)],
         "forward": {"hidden_diff": tensor_diff(o1.last_hidden_state[:,-1:,:], o2.last_hidden_state[:,-1:,:]),
                     "logits_diff": tensor_diff(p1, p2),
                     "official_t0": int(p1.argmax(dim=-1).item()), "engine_t0": int(p2.argmax(dim=-1).item())}}
    out = Path(args.out_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2))
    for k, v in s["component_diffs"].items(): print(f"  {k}:", json.dumps(v, ensure_ascii=False))
    print("  prefill_diff:", json.dumps(s["prefill_diff"], ensure_ascii=False))
    print(f"  token0: official={s['forward']['official_t0']} engine={s['forward']['engine_t0']}")
    print("  saved:", out)

# -- mode: compare-paths ----------------------------------------------------

def _run_compare_paths(args):
    from engine.backend.prefill import OFFICIAL_ASSISTANT_FMT, TaskType
    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = torch.device("cuda:0")
    model = _load_model(model_dir)
    builder, weights, tok = _load_builder(model_dir, str(REPO_ROOT / "workspace/exported/custom-1.7b/weights"), device)
    at = OFFICIAL_ASSISTANT_FMT.format(text=args.text) if args.prompt_mode == "supported" \
         else f"<|im_start|>assistant\n{args.text}<|im_end|>"
    ids = torch.as_tensor(tok(at, return_tensors="pt")["input_ids"], device=device, dtype=torch.long)
    if ids.dim() == 1: ids = ids.unsqueeze(0)
    off_pf, off_tr = build_prefill_like_official(model, ids, args.language, args.speaker, device)
    text_ids = builder._encode_text_ids(args.text)
    plan = builder.build_plan_from_ids(task_type=TaskType.CUSTOM_VOICE, token_ids=text_ids,
                                       language=args.language, speaker=args.speaker, include_eos=True)
    eng_pf = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    eng_tr = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]
    print(f"prompt_mode: {args.prompt_mode}")
    print(f"official: prefill_len={off_pf.shape[1]} trailing_len={len(off_tr)}")
    print(f"engine:   prefill_len={eng_pf.shape[1]} trailing_len={len(eng_tr)}")
    print("prefill_diff:", json.dumps(tensor_diff(off_pf, eng_pf), ensure_ascii=False))
    for i in range(min(len(off_tr), len(eng_tr))):
        print(f"trailing_diff[{i}]:", json.dumps(tensor_diff(off_tr[i], eng_tr[i]), ensure_ascii=False))
    if len(off_tr) != len(eng_tr): print(f"trailing_len_mismatch: {len(off_tr)} vs {len(eng_tr)}")

# -- mode: manual-rollout ---------------------------------------------------

def _sample_token(logits, hist, *, top_k, temperature, rep_pen, suppress):
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper)
    s = logits.float().clone()
    if suppress: s[..., suppress] = -1e9
    if rep_pen != 1.0: s = RepetitionPenaltyLogitsProcessor(rep_pen)(hist, s)
    if temperature != 1.0: s = TemperatureLogitsWarper(temperature)(hist, s)
    if top_k > 0: s = TopKLogitsWarper(top_k)(hist, s)
    return torch.multinomial(torch.softmax(s, dim=-1), num_samples=1).squeeze(-1)

def _run_manual_rollout(args):
    from engine.backend.prefill import OFFICIAL_ASSISTANT_FMT, TaskType
    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    model = _load_model(model_dir); device = next(model.parameters()).device
    talker = model.talker; cp = talker.code_predictor
    builder, weights, tok = _load_builder(model_dir, str(REPO_ROOT / "workspace/exported/custom-1.7b/weights"), device)
    text_ids = builder._encode_text_ids(args.text)
    plan = builder.build_plan_from_ids(task_type=TaskType.CUSTOM_VOICE, token_ids=text_ids,
                                       language=args.language, speaker=args.speaker, include_eos=True)
    pf = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    trailing = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]
    pad_e = weights.tts_pad_embed.to(device=device, dtype=torch.bfloat16)
    suppress = [i for i in range(talker.config.vocab_size-1024, talker.config.vocab_size)
                if i not in (talker.config.codec_eos_token_id,)]
    # manual rollout
    torch.manual_seed(args.seed)
    with torch.no_grad(): out = talker.model(inputs_embeds=pf, use_cache=True, return_dict=True)
    kv = out.past_key_values; ph = out.last_hidden_state[:, -1:, :]
    logits = talker.codec_head(out.last_hidden_state)[:, -1, :]
    hist = torch.empty((1,0), device=device, dtype=torch.long)
    c0 = _sample_token(logits, hist, top_k=50, temperature=0.9, rep_pen=1.05, suppress=suppress)
    hist = torch.cat([hist, c0.unsqueeze(1)], dim=1)
    seq, phases = [c0.item()], ["prefill"]
    for step in range(args.max_steps - 1):
        with torch.no_grad():
            cp_in = torch.cat((ph, talker.model.codec_embedding(c0).unsqueeze(1)), dim=1)
            pred = cp.generate(inputs_embeds=cp_in, max_new_tokens=talker.config.num_code_groups-1,
                               do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                               output_hidden_states=True, return_dict_in_generate=True)
            cpt = pred.sequences
            ch = torch.cat([talker.get_input_embeddings()(c0.unsqueeze(1))] +
                           [cp.get_input_embeddings()[i](cpt[..., i:i+1])
                            for i in range(talker.config.num_code_groups-1)], dim=1)
            ni = ch.sum(1, keepdim=True) + (trailing[step] if step < len(trailing) else pad_e)
            so = talker.model(inputs_embeds=ni, past_key_values=kv, use_cache=True, return_dict=True)
        kv = so.past_key_values; ph = so.last_hidden_state[:, -1:, :]
        logits = talker.codec_head(ph)[:, -1, :]
        c0 = _sample_token(logits, hist, top_k=50, temperature=0.9, rep_pen=1.05, suppress=suppress)
        hist = torch.cat([hist, c0.unsqueeze(1)], dim=1)
        seq.append(c0.item()); phases.append("text" if step < len(trailing) else "pad")
        if c0.item() == talker.config.codec_eos_token_id: break
    # official rollout
    at = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    ids = torch.as_tensor(tok(at, return_tensors="pt")["input_ids"], device=device, dtype=torch.long)
    if ids.dim() == 1: ids = ids.unsqueeze(0)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        cl, _ = model.generate(input_ids=[ids], languages=[args.language], speakers=[args.speaker],
                               non_streaming_mode=False, do_sample=True, subtalker_dosample=True,
                               top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05, max_new_tokens=args.max_steps)
    off = [r[0] for r in cl[0].detach().cpu().tolist()]
    n = min(len(off), len(seq)); div = next((i for i in range(n) if off[i] != seq[i]), -1)
    print(f"trailing_len: {len(trailing)}  official_len: {len(off)}  manual_len: {len(seq)}  first_div: {div}")
    if div >= 0:
        lo, hi = max(0, div-5), min(n, div+10)
        for i in range(lo, hi):
            mk = "<<" if i == div else "  "
            print(f"{mk} step={i:03d} phase={phases[i] if i<len(phases) else 'n/a':7s} off={off[i]:4d} man={seq[i]:4d}")
    else: print("Sequences match on common prefix.")

# -- mode: cp-parity --------------------------------------------------------

def _run_cp_parity(args):
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    from utils import CodePredictorUnrolled
    from engine.backend.prefill import TaskType

    device = torch.device(args.device)
    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    talker = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=args.device, attn_implementation="eager")
    talker.eval(); cp = talker.code_predictor
    cpu = CodePredictorUnrolled(cp, talker.model.codec_embedding, logits_topk=args.top_k).to(device).eval()
    builder, weights, tok = _load_builder(model_dir, str(REPO_ROOT / "workspace/exported/custom-1.7b/weights"), device)
    text_ids = builder._encode_text_ids(args.text)
    plan = builder.build_plan_from_ids(task_type=TaskType.CUSTOM_VOICE, token_ids=text_ids,
                                       language=args.language, speaker=args.speaker, include_eos=True)
    pf = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad(): out = talker.model(inputs_embeds=pf, use_cache=True, return_dict=True)
    ph = out.last_hidden_state[:, -1:, :]; tl = talker.codec_head(out.last_hidden_state)[:, -1, :]
    c0 = tl.argmax(dim=-1)
    print(f"=== CP sampled parity ===  text={args.text}  codec0={c0.item()}  trailing={len(plan.trailing)}")
    # greedy
    with torch.no_grad():
        og = cp.generate(inputs_embeds=torch.cat((ph, talker.model.codec_embedding(c0).unsqueeze(1)), dim=1),
                         max_new_tokens=talker.config.num_code_groups-1, do_sample=False,
                         top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                         output_hidden_states=True, return_dict_in_generate=True).sequences[0].cpu().tolist()
        ug = cpu(ph, c0, cp_gumbel_noise=None,
                 temperature=torch.full((1,1), args.temperature, device=device, dtype=torch.float32)
                 )[0].cpu().tolist()
    print(f"Greedy: off={og} unr={ug} match={og==ug}")
    # sampled
    full = 0; pfx = []; oc = [Counter() for _ in range(15)]; uc = [Counter() for _ in range(15)]
    for seed in range(args.trials):
        torch.manual_seed(seed)
        with torch.no_grad():
            o = cp.generate(inputs_embeds=torch.cat((ph, talker.model.codec_embedding(c0).unsqueeze(1)), dim=1),
                            max_new_tokens=talker.config.num_code_groups-1, do_sample=True,
                            top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                            output_hidden_states=True, return_dict_in_generate=True).sequences[0].cpu().tolist()
        g = torch.Generator(device=device); g.manual_seed(seed)
        gn = -torch.log(-torch.log(torch.rand(1, cpu.num_stages, args.top_k, device=device, dtype=torch.float32,
                                               generator=g).clamp(1e-8,1.0)))
        with torch.no_grad():
            u = cpu(ph, c0, cp_gumbel_noise=gn,
                    temperature=torch.full((1,1), args.temperature, device=device, dtype=torch.float32)
                    )[0].cpu().tolist()
        if o == u: full += 1
        lcp = next((i for i, (a,b) in enumerate(zip(o,u)) if a!=b), min(len(o), len(u)))
        pfx.append(lcp)
        for i,t in enumerate(o): oc[i][t] += 1
        for i,t in enumerate(u): uc[i][t] += 1
        print(f"seed={seed:02d} lcp={lcp:2d} off={o} unr={u}")
    print(f"\nexact match: {full}/{args.trials}  avg lcp: {sum(pfx)/len(pfx):.2f}")

# -- CLI --------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Unified prefill comparison tool")
    p.add_argument("--mode", required=True,
                   choices=["official", "live-vs-exported", "compare-paths", "manual-rollout", "cp-parity"])
    p.add_argument("--text", required=True)
    p.add_argument("--speaker", default="vivian"); p.add_argument("--language", default="auto")
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--out-json", default=None)
    p.add_argument("--prompt-mode", choices=("supported", "raw"), default="supported")
    p.add_argument("--seed", type=int, default=1234); p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--top-k", type=int, default=50); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--trials", type=int, default=20); p.add_argument("--device", default="cuda:0")
    return p

def main(argv=None) -> int:
    _print_deprecation()
    args = build_parser().parse_args(argv)
    if args.out_json is None:
        args.out_json = str(REPO_ROOT / "workspace" / "live_vs_exported_prefill.json")
    {"official": lambda: print("Use --mode compare-paths or --mode manual-rollout for comparison."),
     "live-vs-exported": lambda: _run_live_vs_exported(args),
     "compare-paths": lambda: _run_compare_paths(args),
     "manual-rollout": lambda: _run_manual_rollout(args),
     "cp-parity": lambda: _run_cp_parity(args),
    }[args.mode]()
    return 0

if __name__ == "__main__":
    sys.exit(main())
