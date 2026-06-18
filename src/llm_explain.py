import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _chat_endpoint(base_url: str) -> str:
    """兼容两种写法：SJTU_API_BASE=https://xxx/v1 或 https://xxx/v1/chat/completions。"""
    base_url = (base_url or "").strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _get_api_config() -> Tuple[Optional[str], Optional[str], Optional[str], float]:
    api_key = os.getenv("SJTU_API_KEY")
    base_url = os.getenv("SJTU_API_BASE")
    model_name = os.getenv("SJTU_MODEL_NAME")
    try:
        timeout = float(os.getenv("SJTU_API_TIMEOUT", "30"))
    except ValueError:
        timeout = 30.0
    return api_key, base_url, model_name, timeout


def call_chat_completion(messages: List[Dict[str, str]], temperature: float = 0.2, response_format: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """调用 OpenAI-compatible Chat Completions 接口。返回 (content, error)。"""
    api_key, base_url, model_name, timeout = _get_api_config()
    if not api_key or not base_url or not model_name:
        return None, "缺少 SJTU_API_KEY / SJTU_API_BASE / SJTU_MODEL_NAME 配置"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    try:
        response = requests.post(_chat_endpoint(base_url), headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return str(content).strip(), None
    except Exception as exc:
        return None, str(exc)


def _compact_features(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "无明显高贡献特征"
    parts = []
    for x in items[:8]:
        feature = str(x.get("feature", ""))
        contribution = float(x.get("contribution", 0.0))
        parts.append(f"{feature}，贡献值{contribution:.4f}")
    return "；".join(parts)


def build_llm_explanation_prompt(text, label_name, confidence, support_rumor_detail, support_non_rumor_detail, decision_score, local_explanation):
    return f"""
你是一个谣言检测系统的解释生成模块。
请根据“最终分类结果”和“本地模型证据”生成一段客观、简洁、可复现的中文判断依据。
要求：
1. 不要凭空添加原文没有的信息；
2. 说明哪些词或短语支持“谣言”；
3. 说明哪些词或短语支持“非谣言”；
4. 说明为什么系统最终更偏向当前类别；
5. 字数控制在 100 到 180 字。

原始文本：{text}
最终判断：{label_name}
最终置信度：{confidence:.4f}
支持“谣言”的词或短语及贡献值：{_compact_features(support_rumor_detail)}
支持“非谣言”的词或短语及贡献值：{_compact_features(support_non_rumor_detail)}
本地模型决策分数：{decision_score:.4f}
本地解释草稿：{local_explanation}
请生成最终判断依据：
""".strip()


def generate_llm_or_fallback_explanation(text, label_name, confidence, support_rumor_detail, support_non_rumor_detail, decision_score, local_explanation, use_llm=False):
    if not use_llm:
        return local_explanation, "local"
    prompt = build_llm_explanation_prompt(
        text, label_name, confidence, support_rumor_detail, support_non_rumor_detail, decision_score, local_explanation
    )
    content, error = call_chat_completion([
        {"role": "system", "content": "你负责把谣言检测系统的结构化证据改写成自然语言解释。"},
        {"role": "user", "content": prompt},
    ], temperature=0.2)
    if content:
        return content, "llm"
    return local_explanation, f"local_fallback({error})"


def _extract_json_object(text: str) -> Dict[str, Any]:
    """从大模型输出中尽量提取 JSON 对象，兼容 ```json ... ``` 包裹。"""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("no json object found")
    return json.loads(match.group(0))


def build_llm_judge_prompt(text: str, local_result: Dict[str, Any]) -> str:
    """让大模型作为第二判别器参与判断。"""
    local_label = local_result.get("local_label_name", local_result.get("label_name", "未知"))
    local_conf = float(local_result.get("local_confidence", local_result.get("confidence", 0.0)))
    local_score = float(local_result.get("decision_score", 0.0))
    rumor_features = local_result.get("support_rumor_detail", [])
    non_features = local_result.get("support_non_rumor_detail", [])
    matched_rule = local_result.get("matched_rule", "")

    return f"""
你是一个谣言检测辅助判别器。请你独立阅读文本，并参考本地机器学习模型的证据，判断文本本身更像“谣言”还是“非谣言/辟谣/事实陈述”。

标签定义：
0 = 非谣言：包括官方通报、事实陈述、辟谣澄清、提醒不要传播谣言、声明某传言为假等。
1 = 谣言：包括未经证实的可疑主张、耸动传播、缺少来源的病毒式传言、夸大或明显不可靠的说法等。

重要规则：
- 如果文本是在说明“某个网传消息是假的/已被否认/无证据/不要传播”，通常应判断为 0。
- 如果文本本身在传播未经证实的可疑说法，通常应判断为 1。
- 你可以与本地模型不同，但必须给出简短理由。
- 只输出 JSON，不要输出 Markdown，不要加代码块。

原始文本：{text}

本地模型结果：
- 本地判断：{local_label}
- 本地置信度：{local_conf:.4f}
- 本地决策分数：{local_score:.4f}
- 本地支持谣言特征：{_compact_features(rumor_features)}
- 本地支持非谣言特征：{_compact_features(non_features)}
- 命中的规则：{matched_rule if matched_rule else "无"}

请输出以下 JSON 格式：
{{
  "label": 0 或 1,
  "label_name": "非谣言" 或 "谣言",
  "confidence": 0.0 到 1.0 之间的小数,
  "rationale": "一句话说明判断依据",
  "evidence_rumor": ["支持谣言的证据1", "支持谣言的证据2"],
  "evidence_non_rumor": ["支持非谣言的证据1", "支持非谣言的证据2"],
  "disagree_with_local": true 或 false
}}
""".strip()


def call_llm_judge(text: str, local_result: Dict[str, Any]) -> Dict[str, Any]:
    """调用大模型参与判断。失败时返回 ok=False，不影响本地模型。"""
    prompt = build_llm_judge_prompt(text, local_result)
    content, error = call_chat_completion([
        {"role": "system", "content": "你是一个严谨的谣言检测辅助判别器。你必须只输出 JSON。"},
        {"role": "user", "content": prompt},
    ], temperature=0.1, response_format={"type": "json_object"})

    if not content:
        return {"ok": False, "error": error or "empty response", "raw_output": ""}

    try:
        data = _extract_json_object(content)
        label = int(data.get("label"))
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        label_name = "谣言" if label == 1 else "非谣言"
        return {
            "ok": True,
            "label": label,
            "label_name": label_name,
            "confidence": confidence,
            "rationale": str(data.get("rationale", "")).strip(),
            "evidence_rumor": data.get("evidence_rumor", []) if isinstance(data.get("evidence_rumor", []), list) else [],
            "evidence_non_rumor": data.get("evidence_non_rumor", []) if isinstance(data.get("evidence_non_rumor", []), list) else [],
            "disagree_with_local": bool(data.get("disagree_with_local", False)),
            "raw_output": content,
            "error": "",
        }
    except Exception as exc:
        return {"ok": False, "error": f"parse failed: {exc}", "raw_output": content}
