import re


def clean_text(text, mode=0):
    text = str(text)
    text = text.replace("&amp;", " and ").replace("&lt;", " < ").replace("&gt;", " > ")
    if mode in (0, 1, 2):
        text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
        text = re.sub(r"@\w+", " USER ", text)
    if mode == 0:
        text = re.sub(r"#(\w+)", r" HASHTAG \1 ", text)
    elif mode == 1:
        text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"找不到字段。候选字段：{candidates}，当前列名：{list(df.columns)}")
