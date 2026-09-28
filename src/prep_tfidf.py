# TF-IDF по объявлениям и запросам, все складываем в cache/tfidf.pkl
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CACHE, ITEMS, TRAIN, norm

# из train берем только то, что нужно для prior по подкатегориям и для валидации
TRAIN_COLUMNS = [
    "search_query", "search_location_id", "search_infm_params_text",
    "item_id", "item_microcat_id", "item_location_id",
]

# что сохраняем про объявления для следующих шагов
ITEM_COLUMNS = [
    "item_id", "item_location_id", "item_microcat_id",
    "lat", "lon", "t_title", "t_par",
]


def clean(column: pd.Series) -> pd.Series:
    # пустые значения превращаем в пустую строку, иначе векторизатор упадет на NaN
    return column.fillna("").map(norm)


def main() -> None:
    items = pd.read_parquet(ITEMS)
    items["t_title"] = clean(items.item_title_raw)
    items["t_par"] = clean(items.item_infm_params_text)
    # описание обрезаем до 400 символов: главное обычно в начале, а длинные тексты сильно раздувают словарь
    items["t_desc"] = clean(items.item_description_raw.str[:400])
    # координаты хранятся как строки, некорректные значения становятся NaN
    items["lat"] = pd.to_numeric(items.item_latitude, errors="coerce")
    items["lon"] = pd.to_numeric(items.item_longitude, errors="coerce")

    train = pd.read_parquet(TRAIN, columns=TRAIN_COLUMNS)
    train["q"] = clean(train.search_query)
    # словарь учим и на запросах, чтобы в нем были слова, которыми люди ищут
    queries = train.q.drop_duplicates()

    # char n-граммы нормально переживают падежи и опечатки
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=600_000, sublinear_tf=True, dtype=np.float32)
    char_vec.fit(pd.concat([items.t_title, queries]))

    # словесные 1-2-граммы ловят точные совпадения фраз вроде ремонт двигателя
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32, token_pattern=r"\b\w\w+\b")
    word_vec.fit(pd.concat([items.t_title, items.t_desc, queries]))

    # ct и wt - только заголовок, wd - заголовок с параметрами и описанием, cp - параметры объявления, их сравниваем с фильтрами запроса
    full_text = items.t_title + " " + items.t_par + " " + items.t_desc
    matrices = {
        "ct": char_vec.transform(items.t_title),
        "cp": char_vec.transform(items.t_par),
        "wt": word_vec.transform(items.t_title),
        "wd": word_vec.transform(full_text),
    }

    # векторизаторы тоже сохраняем, чтобы в features.py переводить запросы в тот же словарь
    cache = dict(it=items[ITEM_COLUMNS], X=matrices,
                 vc=char_vec, vw=word_vec, tr=train)
    with open(CACHE / "tfidf.pkl", "wb") as f:
        pickle.dump(cache, f, protocol=5)

    print({name: m.shape for name, m in matrices.items()})


if __name__ == "__main__":
    main()
