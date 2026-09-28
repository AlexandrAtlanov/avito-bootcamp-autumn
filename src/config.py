# Общие настройки проекта: пути к данным и кешу, размер ответа и нормализация текста

import re
from pathlib import Path

# все пути считаются от корня проекта, поэтому скрипты работают из любой папки
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
SUBMISSIONS = ROOT / "submissions"

TRAIN = DATA / "train.parquet"
ITEMS = DATA / "benchmark_items.parquet"
QUERIES = DATA / "benchmark_queries.parquet"

# сколько кандидатов отдаем на каждый запрос, столько требует формат ответа
TOP_K = 50

# папку для кеша создаем сразу, чтобы после git clone ничего не создавать руками
CACHE.mkdir(exist_ok=True)


def norm(text: object) -> str:
    # нижний регистр, одна буква е на все варианты написания,
    # все кроме букв и цифр заменяем пробелом
    text = str(text).lower().replace("ё", "е")
    
    return re.sub(r"[^0-9a-zа-я ]", " ", text)
