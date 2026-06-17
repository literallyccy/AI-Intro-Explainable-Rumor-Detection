import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def build_llm_prompt(
    text,
    label_name,
    confidence,
    support_rumor_detail,
    support_non_rumor_detail,
    decision_score,
    local_explanation
):
    def compact(items):
        if not items:
            return "无明显高贡献特征"
        return "；".join([
            f"{x['feature']}，贡献值{x['contribution']:.4f}"
            for x in items
        ])

    return f"""
你是一个谣言检测系统的解释生成模块。
请根据“本地分类模型”的输出生成一段客观、简洁、可复现的中文判断依据。

重要要求：
1. 不要改变本地模型的最终判断；
2. 不要凭空添加原文没有的信息；
3. 需要说明哪些词或片段支持“谣言”；
4. 需要说明哪些词或片段支持“非谣言”；
5. 需要说明为什么模型最终更偏向当前类别；
6. 字数控制在 100 到 180 字。

原始文本：
{text}

本地模型最终判断：
{label_name}

本地模型置信度：
{confidence:.4f}

支持“谣言”的词或片段及贡献值：
{compact(support_rumor_detail)}

支持“非谣言”的词或片段及贡献值：
{compact(support_non_rumor_detail)}

最终决策分数：
{decision_score:.4f}

本地解释草稿：
{local_explanation}

请生成最终判断依据：
""".strip()


def call_llm_explanation(prompt):
    """
    调用 OpenAI-compatible Chat Completions API。
    适用于学校提供的兼容接口。

    通过环境变量配置：
        SJTU_API_KEY
        SJTU_API_BASE
        SJTU_MODEL_NAME

    如果未配置或调用失败，返回 None，主程序自动退回本地解释。
    """
    api_key = os.getenv("SJTU_API_KEY")
    base_url = os.getenv("SJTU_API_BASE")
    model_name = os.getenv("SJTU_MODEL_NAME")

    if not api_key or not base_url or not model_name:
        return None

    base_url = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你负责把谣言检测模型的结构化证据改写成自然语言解释。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def generate_llm_or_fallback_explanation(
    text,
    label_name,
    confidence,
    support_rumor_detail,
    support_non_rumor_detail,
    decision_score,
    local_explanation,
    use_llm=False
):
    if not use_llm:
        return local_explanation, "local"

    prompt = build_llm_prompt(
        text=text,
        label_name=label_name,
        confidence=confidence,
        support_rumor_detail=support_rumor_detail,
        support_non_rumor_detail=support_non_rumor_detail,
        decision_score=decision_score,
        local_explanation=local_explanation
    )

    llm_text = call_llm_explanation(prompt)
    if llm_text:
        return llm_text, "llm"
    return local_explanation, "local_fallback"
