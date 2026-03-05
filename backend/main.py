from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_PATH = Path(__file__).resolve().parent.parent / "questions.json"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# -----------------------
# Data
# -----------------------
def load_data() -> Dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------
# Rule-based scoring
# -----------------------
def normalize(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("＝", "=").replace("＜", "<").replace("＞", ">")
    s = re.sub(r"\s+", "", s)
    return s


def hit_any(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if p and p in text:
            return True
    return False


def score_answer_rule(answer_text: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
    raw = answer_text or ""
    t = normalize(raw)

    details = {
        "structure": [],
        "keywords": {"hit": [], "miss": []},
        "conclusion_hit": False,
    }

    total = 0

    # Structure scoring
    for item in rubric.get("structure", []):
        name = item["name"]
        pts = int(item["points"])
        patterns = item.get("patterns", [])
        ok = hit_any(t, [normalize(x) for x in patterns])
        details["structure"].append({"name": name, "ok": ok, "points": pts if ok else 0})
        total += pts if ok else 0

    # Keyword scoring (cap)
    kw = rubric.get("keywords", {})
    terms = kw.get("terms", [])
    max_points = int(kw.get("max_points", 0))
    hit = []
    miss = []
    for term in terms:
        if normalize(term) in t:
            hit.append(term)
        else:
            miss.append(term)
    details["keywords"]["hit"] = hit
    details["keywords"]["miss"] = miss

    kw_points = min(len(hit), max_points)
    total += kw_points

    # Conclusion
    concl = rubric.get("conclusion", {})
    concl_patterns = [normalize(x) for x in concl.get("patterns", [])]
    concl_points = int(concl.get("points", 0))
    concl_hit = hit_any(t, concl_patterns)
    details["conclusion_hit"] = concl_hit
    total += concl_points if concl_hit else 0

    total = min(total, 10)

    feedback = []
    for sitem in details["structure"]:
        if not sitem["ok"]:
            feedback.append(f"缺少结构要素：{sitem['name']}")
    if len(hit) == 0 and len(terms) > 0:
        feedback.append("建议加入数学关键词（如：{}）".format("、".join(terms[:3])))
    if concl_patterns and not concl_hit:
        feedback.append("结论可能不够明确（可写出谁更大/更小/是否相等）")

    return {"score": total, "details": details, "feedback": feedback}


# -----------------------
# LLM scoring (OpenAI-compatible HTTP)
# -----------------------
def _llm_base() -> str:
    """
    Prefer LLM_BASE_URL for hosted providers, fallback to OLLAMA_BASE_URL.
    Supports:
      LLM_BASE_URL=https://api.example.com
      LLM_BASE_URL=https://api.example.com/v1
      OLLAMA_BASE_URL=http://localhost:11434
      OLLAMA_BASE_URL=http://localhost:11434/v1
    """
    base = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    ).rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _llm_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_json_block(text: str) -> str:
    """
    Extract JSON from model output.
    Handles:
      - pure JSON
      - ```json ... ```
      - ``` ... ```
    """
    t = (text or "").strip()
    if t.startswith("```"):
        # remove first fence line
        lines = t.splitlines()
        if len(lines) >= 2:
            # drop first line (``` or ```json)
            lines = lines[1:]
            # drop last fence line if exists
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        # If it started with ```json, first char could be json label already removed above
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def score_answer_ollama(stem: str, reference_answer: str, rubric: dict, answer_text: str) -> dict:
    """
    Call OpenAI-compatible chat/completions endpoint via plain HTTP.
    """
    base = _llm_base()
    url = f"{base}/chat/completions"
    model = os.getenv("LLM_MODEL", "qwen2.5:7b")

    system_prompt = (
        "你是一名小学数学阅卷老师。只能依据学生答案评分，不能脑补。"
        "严格按 rubric 评分，输出必须是【纯JSON】，不要输出任何额外文字。"
    )

    # 给一点“更像机器评分器”的约束，减少跑偏
    user_prompt = f"""
请只输出 JSON（不要任何解释文字），字段必须齐全：

{{
  "score": 0-10 的整数,
  "decision": "同意" 或 "不同意" 或 "不确定",
  "strengths": ["..."],
  "issues": ["..."],
  "missing_points": ["..."],
  "suggestion": "...",
  "reason": "一句话评分理由",
  "confidence": 0 到 1 的小数
}}

【题干】
{stem}

【参考答案】
{reference_answer}

【rubric】
{rubric}

【学生答案】
{answer_text}
""".strip()

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(url, json=payload, headers=_llm_headers(), timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    text = _extract_json_block(content)

    def _parse_or_raise(txt: str) -> dict:
        try:
            obj = json.loads(txt)
        except Exception as e:
            raise RuntimeError(f"LLM输出不是合法JSON: {e}; raw={txt[:300]}")
        required = ["score","decision","strengths","issues","missing_points","suggestion","confidence"]
        for k in required:
            if k not in obj:
                raise RuntimeError(f"LLM输出缺少字段 {k}; raw={txt[:300]}")
        obj["score"] = max(0, min(10, int(obj["score"])))
        obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
        reason = str(obj.get("reason", "")).strip()
        if not reason:
            issues = obj.get("issues") or []
            strengths = obj.get("strengths") or []
            suggestion = str(obj.get("suggestion", "")).strip()
            if issues:
                reason = f"主要失分点：{issues[0]}"
            elif strengths:
                reason = f"得分依据：{strengths[0]}"
            elif suggestion:
                reason = suggestion[:80]
            else:
                reason = "答案与评分细则匹配度一般。"
        obj["reason"] = reason
        return obj

    # 一次解析
    try:
        return _parse_or_raise(text)
    except Exception:
        # 轻量重试：再请求一次，强制它只输出JSON（本地模型常见）
        retry_prompt = (
            "你刚才没有按要求输出纯JSON。"
            "请立刻重新输出【纯JSON】且不要任何其它文字。"
        )
        payload2 = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": content},
                {"role": "user", "content": retry_prompt},
            ],
        }
        resp2 = requests.post(url, json=payload2, headers=_llm_headers(), timeout=90)
        if resp2.status_code != 200:
            raise RuntimeError(f"LLM HTTP {resp2.status_code}: {resp2.text[:300]}")
        data2 = resp2.json()
        content2 = data2["choices"][0]["message"]["content"]
        text2 = _extract_json_block(content2)
        return _parse_or_raise(text2)


# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="MW答题系统 Demo", version="0.2.0")


class ScoreReq(BaseModel):
    question_id: str
    answer_text: str


@app.get("/api/lessons")
def get_lessons():
    data = load_data()
    return sorted(data["lessons"], key=lambda x: x.get("order", 0))


@app.get("/api/questions")
def get_questions(lesson_id: Optional[str] = None):
    data = load_data()
    qs = data["questions"]
    if lesson_id:
        qs = [q for q in qs if q["lesson_id"] == lesson_id]
    return qs


@app.get("/api/llm-health")
def llm_health():
    """
    Quick sanity check for LLM endpoint + model.
    """
    try:
        base = _llm_base()
        url = f"{base}/chat/completions"
        model = os.getenv("LLM_MODEL", "qwen2.5:7b")
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": "只输出JSON：{\"ok\":true}"}],
        }
        resp = requests.post(url, json=payload, headers=_llm_headers(), timeout=30)
        return {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "base": base,
            "model": model,
            "hint": resp.text[:200],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/score")
def post_score(req: ScoreReq, mode: str = Query(default="rule", pattern="^(rule|llm|hybrid)$")):
    data = load_data()
    q = next((x for x in data["questions"] if x["id"] == req.question_id), None)
    if not q:
        raise HTTPException(status_code=404, detail="question not found")

    rule_out = score_answer_rule(req.answer_text, q["rubric"])

    if mode == "rule":
        return {"mode": "rule", **rule_out}

    try:
        llm_out = score_answer_ollama(
            stem=q["stem"],
            reference_answer=q["reference_answer"],
            rubric=q["rubric"],
            answer_text=req.answer_text,
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "mode": mode,
                "error": "LLM评分失败",
                "message": str(e),
                "rule_fallback": rule_out,
            },
        )

    if mode == "llm":
        return {"mode": "llm", "llm": llm_out, "rule": rule_out}

    # hybrid
    rule_score = float(rule_out.get("score", 0))
    llm_score = float(llm_out.get("score", 0))
    final_score = round(rule_score * 0.4 + llm_score * 0.6, 1)
    needs_review = abs(rule_score - llm_score) >= 4

    return {
        "mode": "hybrid",
        "rule": rule_out,
        "llm": llm_out,
        "final_score": final_score,
        "needs_review": needs_review,
    }


# Serve frontend
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
