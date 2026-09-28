# Ранжирование кандидатов и сабмит
import re
import sys
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from config import CACHE, ITEMS, SUBMISSIONS, TOP_K

# любая из четырех моделей ансамбля
Model = HistGradientBoostingClassifier | lgb.LGBMRanker | lgb.LGBMClassifier

# простые линейные смеси признаков, нужны только для сравнения с бустингами
TFIDF = dict(ct=1, wt=0.5, wd=0.5, cp=0.3)
GEO = dict(loc=0.6, near=0.3)
BASELINES = {
    "tfidf_title_char": dict(ct=1),
    "tfidf_words_desc": dict(wd=1),
    "tfidf_all": TFIDF,
    "tfidf_all+loc": TFIDF | GEO,
    "tfidf_all+loc+prior": TFIDF | GEO | dict(pr=1),
    "+emb": TFIDF | GEO | dict(pr=1, emb=1),
}

# служебные колонки, модели их не видят
NOT_FEATURES = {"i", "qi", "item_id", "query_id", "y"}

# свойства самого объявления, берем прямо из корпуса
ITEM_ATTRS = [
    "item_price", "item_rating", "item_rating_reviews_count",
    "item_is_phone_hidden", "item_is_message_forbidden", "item_category_id",
]


def load(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    attrs = pd.read_parquet(ITEMS, columns=["item_id", *ITEM_ATTRS])
    attrs = attrs.astype({c: float for c in ITEM_ATTRS})

    df = pd.read_parquet(CACHE / f"feats_{mode}.parquet")
    df = df.merge(attrs, on="item_id", how="left")
    # near: около 1 рядом с центром локации запроса и около 0 далеко от него
    df["near"] = np.exp(-df.dist / 30)

    # то же, но относительно других кандидатов этого запроса
    # у разных запросов значения признаков несравнимы, а ранг внутри запроса
    # и отставание от лучшего кандидата сравнимы всегда
    by_query = df.groupby("qi")

    for name in ["ct", "wt", "wd", "pr", "emb"]:
        if name in df:
            df[name + "_rk"] = by_query[name].rank(ascending=False, method="min")
            df[name + "_gap"] = by_query[name].transform("max") - df[name]

    df["dist_rk"] = by_query.dist.rank(method="min")
    # сколько кандидатов запроса лежит в его локации
    df["n_loc"] = by_query["loc"].transform("sum")

    return df, pd.read_pickle(CACHE / f"queries_{mode}.pkl")


# TOP_K кандидатов с наибольшим скором для каждого запроса
def top_k(df: pd.DataFrame, score: np.ndarray | pd.Series) -> pd.DataFrame:
    ranked = df[["qi", "item_id"]].assign(score=score)
    ranked = ranked.sort_values(["qi", "score"], ascending=[True, False])

    return ranked.groupby("qi").head(TOP_K)


# доля правильных объявлений запроса, попавших в топ, усредненная по запросам. Так считается метрика соревнования
def recall(df: pd.DataFrame, queries: pd.DataFrame, score: np.ndarray | pd.Series | None = None) -> float:
    # без score считаем полноту всего пула
    y = df.y if score is None else df.y[top_k(df, score).index]
    hits = y.groupby(df.qi).sum().reindex(range(len(queries)), fill_value=0)

    return float(np.mean(hits / queries.rel.map(len)))


# взвешенная сумма признаков, отсутствующие признаки пропускаем
def linear(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(w * df[name] for name, w in weights.items() if name in df)


def columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES]


def sample(df: pd.DataFrame, neg_share: float) -> pd.DataFrame:
    # все позитивы и часть негативов, иначе учится слишком долго
    keep = np.random.RandomState(0).rand(len(df)) < neg_share

    return df[(df.y == 1) | keep]


def with_positives(df: pd.DataFrame) -> pd.DataFrame:
    # запросы без единого позитива ранкеру ничего не дают
    return df[df.groupby("qi").y.transform("max") == 1]


# классификатор: для каждой пары отдельно предсказывает, выберут ли объявление
def fit_gbm(df: pd.DataFrame) -> HistGradientBoostingClassifier:
    train = sample(df, 0.2)

    model = HistGradientBoostingClassifier(
        max_iter=700, 
        learning_rate=0.05, 
        max_leaf_nodes=63,
        min_samples_leaf=100, 
        l2_regularization=1.0, 
        random_state=0,
    )

    return model.fit(train[columns(df)], train.y)


# LambdaRank учится сразу на порядке кандидатов внутри запроса, это ближе всего к нашей метрике
def fit_lambdarank(df: pd.DataFrame) -> lgb.LGBMRanker:
    train = sample(with_positives(df), 0.3)

    model = lgb.LGBMRanker(
        n_estimators=600, 
        learning_rate=0.05, 
        num_leaves=63,
        min_child_samples=50, 
        subsample=0.8, 
        subsample_freq=1,
        colsample_bytree=0.8, 
        random_state=0, verbose=-1,
    )
    # ранкеру нужно знать, сколько строк подряд относятся к одному запросу
    group = train.groupby("qi").size()

    return model.fit(train[columns(df)], train.y, group=group)


# тот же ранкер с другими параметрами и seed: ошибается иначе и этим помогает ансамблю
def fit_lambdarank_deep(df: pd.DataFrame) -> lgb.LGBMRanker:
    train = sample(with_positives(df), 0.3)

    model = lgb.LGBMRanker(
        n_estimators=800, 
        learning_rate=0.05, 
        num_leaves=127,
        min_child_samples=30, 
        subsample=0.7, 
        subsample_freq=1,
        colsample_bytree=0.6, 
        random_state=1, 
        verbose=-1,
    )
    group = train.groupby("qi").size()

    return model.fit(train[columns(df)], train.y, group=group)


# еще один классификатор, уже на LightGBM
def fit_lgb_binary(df: pd.DataFrame) -> lgb.LGBMClassifier:
    train = sample(df, 0.2)

    model = lgb.LGBMClassifier(
        n_estimators=700, 
        learning_rate=0.05, 
        num_leaves=63,
        min_child_samples=100, 
        subsample=0.8, 
        subsample_freq=1,
        colsample_bytree=0.8, 
        random_state=2, 
        verbose=-1,
    )

    return model.fit(train[columns(df)], train.y)


def predict(model: Model, df: pd.DataFrame) -> np.ndarray:
    X = df[columns(df)]

    # у классификаторов берем вероятность класса 1, у ранкеров сырой скор
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    
    return model.predict(X)


def ensemble(df: pd.DataFrame, scores: list[np.ndarray], weights: list[float] | None = None) -> pd.Series:
    # у моделей разные шкалы, поэтому усредняем ранги внутри запроса
    weights = weights or [1] * len(scores)

    ranks = (w * pd.Series(s, index=df.index).groupby(df.qi).rank(pct=True)
             for w, s in zip(weights, scores))
    
    return sum(ranks) / sum(weights)


# веса подбирали на val (0.8749), на лидерборде 0.8258
MODELS: dict[str, tuple[Callable[[pd.DataFrame], Model], float]] = {
    "gbm": (fit_gbm, 2),
    "lambdarank": (fit_lambdarank, 2),
    "lambdarank_deep": (fit_lambdarank_deep, 1),
    "lgb_binary": (fit_lgb_binary, 1),
}


# фолды делим по запросам, чтобы все кандидаты одного запроса попадали в один фолд, иначе оценка завышена
def cross_val(df: pd.DataFrame, fit: Callable[[pd.DataFrame], Model], folds: int = 3) -> np.ndarray:
    fold = df.qi % folds
    score = np.zeros(len(df))

    for k in range(folds):
        score[fold == k] = predict(fit(df[fold != k]), df[fold == k])

    return score


# проверяем формат ответа до сохранения, чтобы не потратить попытку на битый файл
def check(answer: pd.DataFrame, queries: pd.DataFrame) -> None:
    items = set(pd.read_parquet(ITEMS, columns=["item_id"]).item_id)
    assert answer.query_id.is_unique
    assert set(answer.query_id) == set(queries.query_id)

    for row in answer.answer:
        ids = row.split(" ")
        assert 1 <= len(ids) <= TOP_K and len(set(ids)) == len(ids), row
        assert all(re.fullmatch(r"[0-9a-f]{16}", i) and i in items
                   for i in ids), row


# модели учим на всей валидации и предсказываем на запросах бенчмарка
def submit(name: str) -> None:
    val = load("val")[0]
    df, queries = load("test")

    names = list(MODELS) if name == "ensemble" else [name]
    scores = [predict(MODELS[m][0](val), df) for m in names]
    score = ensemble(df, scores, [MODELS[m][1] for m in names])

    answers = top_k(df, score).groupby("qi").item_id.agg(" ".join)
    answer = pd.DataFrame({
        "query_id": queries.query_id,
        "answer": answers.reindex(range(len(queries))),
    })
    check(answer, queries)

    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"answer_{name}.csv"
    answer.to_csv(path, index=False, encoding="utf-8")
    print("saved", path)


# метрики всех моделей на валидации, бустинги через кросс-валидацию
def evaluate() -> None:
    df, queries = load("val")
    per_query = len(df) // len(queries)
    print(f"pool recall {recall(df, queries):.4f}, " f"~{per_query} candidates per query")

    for name, weights in BASELINES.items():
        if name != "+emb" or "emb" in df:
            score = linear(df, weights)
            print(f"{name:22s} {recall(df, queries, score):.4f}", flush=True)

    scores = []
    for name, (fit, _) in MODELS.items():
        scores.append(cross_val(df, fit))
        print(f"{name:22s} {recall(df, queries, scores[-1]):.4f}", flush=True)

    weights = [w for _, w in MODELS.values()]
    score = ensemble(df, scores, weights)
    print(f"{'ensemble':22s} {recall(df, queries, score):.4f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["submit"]:
        submit(sys.argv[2] if len(sys.argv) > 2 else "ensemble")
    else:
        evaluate()
