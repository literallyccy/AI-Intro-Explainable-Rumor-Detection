import re


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\S+", " USER ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"找不到字段。候选字段：{candidates}，当前列名：{list(df.columns)}"
    )
