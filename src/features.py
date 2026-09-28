# Отбор кандидатов и признаки для пар (запрос, объявление)
import pickle
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CACHE, QUERIES, norm

# столько запросов обрабатываем за раз: на каждый признак получается
# матрица 100 x 189k, больше не помещается в память
BATCH = 100
POOL_MAIN = 600   # берем по общему скору
POOL_EXTRA = 150  # добираем по каждому сигналу отдельно
NO_DIST = 5000    # км, если координат нет


# берем n групп так, чтобы доля запросов с фильтрами была как в бенчмарке
def take(groups: pd.DataFrame, n: int, par_share: float = 0.37) -> pd.DataFrame:
    has_par = groups.par != ""
    k = int(n * par_share)

    return pd.concat([groups[has_par].head(k), groups[~has_par].head(n - k)])


def val_queries(train: pd.DataFrame, item_ids: set[str], n: int, seen_share: float = 0.37, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    # В val идут группы (запрос, локация, фильтры) из train, у которых выбранное объявление есть в корпусе. Пропорции подогнаны под бенчмарк
    # около 37% текстов встречаются в train и  примерно 37% запросов с фильтрами
    # Иначе val сильно расходится с лидербордом

    rng = np.random.RandomState(seed)
    key = ["q", "search_location_id", "par"]
    train = train.assign(par=train.search_infm_params_text.fillna(""))

    texts = train.q.unique()
    # 20% текстов целиком убираем из train, для модели такие запросы новые
    unseen = set(rng.choice(texts, len(texts) // 5, replace=False))
    # сколько разных групп у текста: если больше одной, текст останется в train
    # даже после удаления валидационной группы
    groups_per_text = train.drop_duplicates(key).q.value_counts()

    groups = (train[train.item_id.isin(item_ids)]
              .groupby(key).item_id.unique()
              .reset_index()
              .sample(frac=1, random_state=seed))
    
    is_unseen = groups.q.isin(unseen)
    # знакомый текст встречается в train еще в другой локации или с другими
    # фильтрами, как 37% запросов бенчмарка
    can_be_seen = ~is_unseen & groups.q.map(groups_per_text).gt(1)

    n_seen = int(n * seen_share)
    queries = pd.concat([take(groups[is_unseen], n - n_seen), take(groups[can_be_seen], n_seen)])
    
    queries = (queries.set_axis(["q", "loc", "par", "rel"], axis=1)
               .reset_index(drop=True))
    
    queries["query_id"] = [f"val{i:06d}" for i in range(len(queries))]

    # из train убираем сами валидационные группы и все новые тексты,
    # иначе prior и дообучение e5 увидят ответы
    held = set(queries[["q", "loc", "par"]].itertuples(index=False, name=None))

    train_keys = zip(train.q, train.search_location_id, train.par)
    is_held = np.array([k in held for k in train_keys])

    return queries, train[~train.q.isin(unseen) & ~is_held]


# запросы бенчмарка в том же формате, что и валидационные
def test_queries() -> pd.DataFrame:
    raw = pd.read_parquet(QUERIES)

    return pd.DataFrame({
        "query_id": raw.query_id,
        "q": raw.search_query.fillna("").map(norm),
        "loc": raw.search_location_id,
        "par": raw.search_infm_params_text.fillna(""),
    })


class MicrocatPrior:
    # распределение подкатегорий у k самых похожих запросов из train

    def __init__(self, train: pd.DataFrame, vectorizer: TfidfVectorizer, microcats: np.ndarray, k: int = 20) -> None:
        column = pd.Series(np.arange(len(microcats)), index=microcats)
        train = train[train.item_microcat_id.isin(column.index)]
        counts = (train.groupby(["q", "item_microcat_id"]).size().reset_index(name="n"))

        texts = counts.q.unique()
        row = pd.Series(np.arange(len(texts)), index=texts)
        # матрица текст x подкатегория: сколько раз по тексту выбирали
        # объявление из этой подкатегории
        matrix = sp.csr_matrix(
            (counts.n.astype(np.float32),
             (row[counts.q], column[counts.item_microcat_id])),
            shape=(len(texts), len(microcats)),
        )

        # нормируем строки и получаем распределение по подкатегориям
        self.dist = sp.csr_matrix(matrix.multiply(1 / matrix.sum(axis=1)))
        self.texts_t = vectorizer.transform(texts).T.tocsr()
        self.k = k

    def __call__(self, query_vecs: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        # ищем k ближайших текстов из train и смешиваем их распределения,
        # сходство в 4 степени, чтобы почти точные совпадения весили больше всего
        sim = (query_vecs @ self.texts_t).toarray()
        top = np.argpartition(-sim, self.k, axis=1)[:, :self.k]
        weights = np.take_along_axis(sim, top, axis=1) ** 4

        rows = np.repeat(np.arange(len(sim)), self.k)
        knn = sp.csr_matrix((weights.ravel(), (rows, top.ravel())), shape=sim.shape)
        total = weights.sum(axis=1, keepdims=True) + 1e-9
        prior = (knn @ self.dist).toarray() / total

        # второе значение - сходство с самым близким текстом, по нему модель понимает, насколько можно верить prior
        return prior.astype(np.float32), sim.max(axis=1)


class Geo:
    # у запроса нет координат, поэтому за центр локации берем медиану координат объявлений в ней

    def __init__(self, items: pd.DataFrame) -> None:
        centers = (items.groupby("item_location_id")[["lat", "lon"]].median().dropna())

        self.centers = {loc: np.radians(xy) for loc, xy in zip(centers.index, centers.to_numpy())}
        self.lats = np.radians(items.lat.fillna(0).to_numpy())
        self.lons = np.radians(items.lon.fillna(0).to_numpy())
        self.has_coords = items.lat.notna().to_numpy()
        self.item_loc = items.item_location_id.to_numpy()

    # same_loc - объявление в той же локации, что и запрос, dist - расстояние до центра локации запроса
    def features(self, locs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        same_loc = (self.item_loc == locs[:, None]).astype(np.float32)
        dist = np.full(same_loc.shape, NO_DIST, np.float32)

        for r, loc in enumerate(locs):
            if loc in self.centers:
                d = haversine(*self.centers[loc], self.lats, self.lons)
                dist[r] = np.where(self.has_coords, d, NO_DIST)

        return same_loc, dist


# расстояние по поверхности Земли от одной точки до массива точек
def haversine(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    # км, углы в радианах
    cos = (np.sin(lat) * np.sin(lats) + np.cos(lat) * np.cos(lats) * np.cos(lons - lon))
    
    return 6371 * np.arccos(np.clip(cos, -1, 1))


# argpartition быстрее полной сортировки, порядок внутри топа не важен
def top_ids(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(-scores, k)[:k]


def candidate_pool(f: dict[str, np.ndarray]) -> np.ndarray:
    # f - признаки одного запроса против всех объявлений корпуса

    # near плавно падает с расстоянием: на 30 км примерно 0.37
    near = np.exp(-f["dist"] / 30)
    # base - грубая ручная смесь сигналов, по ней берем основную часть пула
    base = (f["ct"] + 0.5 * f["wt"] + 0.5 * f["wd"] + f["pr"] + 0.6 * f["loc"] + 0.3 * near)
    # отдельные топы по каждому сигналу добавляют кандидатов, которых общий
    # скор мог пропустить, например похожие по смыслу объявления из соседней локации
    sources = [f["ct"], f["wd"], f["ct"] + f["loc"], f["pr"] + f["loc"] + 0.3 * f["ct"]]

    if "emb" in f:
        sources += [f["emb"] + 0.1 * f["loc"], f["emb"] + 0.1 * f["loc"] + 0.3 * f["pr"]]

    tops = [top_ids(base, POOL_MAIN)]
    tops += [top_ids(s, POOL_EXTRA) for s in sources]

    return np.unique(np.concatenate(tops))


def query_embeddings(queries: pd.DataFrame) -> np.ndarray:
    # импорт внутри функции, чтобы без torch можно было считать признаки без эмбеддингов
    from embed import encode
    q_emb = encode(queries.q.tolist(), "query: ", max_len=24)

    return q_emb.astype(np.float32)


def main(mode: str, n_val: int) -> None:
    start = time.time()
    with open(CACHE / "tfidf.pkl", "rb") as f:
        cache = pickle.load(f)

    items, train, X = cache["it"], cache["tr"], cache["X"]
    char_vec, word_vec = cache["vc"], cache["vw"]

    if mode == "val":
        queries, train = val_queries(train, set(items.item_id), n_val)
    else:
        queries = test_queries()

    microcats, item_mc = np.unique(items.item_microcat_id, return_inverse=True)
    prior = MicrocatPrior(train, char_vec, microcats)
    # mc_pop - доля подкатегории объявления в train, то есть насколько популярен этот вид услуг вообще
    mc_share = train.item_microcat_id.value_counts(normalize=True)
    mc_pop = (items.item_microcat_id.map(mc_share).fillna(0).to_numpy(np.float32))
    geo = Geo(items)

    # матрицы корпуса транспонируем заранее, так умножение быстрее
    X_t = {name: m.T.tocsr() for name, m in X.items()}
    q_char = char_vec.transform(queries.q)
    q_word = word_vec.transform(queries.q)
    q_par = char_vec.transform(queries.par.map(norm))

    emb_path = CACHE / "emb_items.npy"
    emb = np.load(emb_path).astype(np.float32) if emb_path.exists() else None

    if emb is not None:
        q_emb = query_embeddings(queries)

    parts = []
    for lo in range(0, len(queries), BATCH):
        batch = slice(lo, lo + BATCH)
        # признаки считаем для батча запросов сразу против всего корпуса, потом у каждого запроса оставляем только его пул
        feats = {
            "ct": (q_char[batch] @ X_t["ct"]).toarray(),
            "wt": (q_word[batch] @ X_t["wt"]).toarray(),
            "wd": (q_word[batch] @ X_t["wd"]).toarray(),
            "cp": (q_par[batch] @ X_t["cp"]).toarray(),
        }
        mc_prior, pmax = prior(q_char[batch])
        feats["pr"] = mc_prior[:, item_mc]
        feats["loc"], feats["dist"] = geo.features(queries["loc"].to_numpy()[batch])

        if emb is not None:
            feats["emb"] = q_emb[batch] @ emb.T

        for r in range(len(feats["ct"])):
            row = {name: values[r] for name, values in feats.items()}
            pool = candidate_pool(row)
            part = pd.DataFrame({name: v[pool] for name, v in row.items()})
            part["i"] = pool
            part["qi"] = lo + r
            part["mc_pop"] = mc_pop[pool]
            part["pmax"] = pmax[r]
            parts.append(part)

        done = min(lo + BATCH, len(queries))
        print(f"{done}/{len(queries)}  {time.time() - start:.0f}s", flush=True)

    df = pd.concat(parts, ignore_index=True)
    # float32 вдвое уменьшает файл с признаками
    df = df.astype({c: np.float32 for c in df.columns if df[c].dtype == np.float64})
    df["item_id"] = items.item_id.to_numpy()[df.i]
    df["query_id"] = queries.query_id.to_numpy()[df.qi]

    if mode == "val":
        # на валидации отмечаем правильные ответы, для них y = 1
        relevant = set(queries[["query_id", "rel"]].explode("rel").itertuples(index=False, name=None))
        pairs = zip(df.query_id, df.item_id)
        df["y"] = np.array([p in relevant for p in pairs], dtype=np.int8)

    df.to_parquet(CACHE / f"feats_{mode}.parquet")
    queries.to_pickle(CACHE / f"queries_{mode}.pkl")
    print(f"rows {len(df)}, {time.time() - start:.0f}s")


# после дообучения e5 пул не пересобираем, а только обновляем колонку emb, так в несколько раз быстрее
def refresh_embeddings(mode: str, chunk: int = 500_000) -> None:
    df = pd.read_parquet(CACHE / f"feats_{mode}.parquet")
    queries = pd.read_pickle(CACHE / f"queries_{mode}.pkl")
    emb = np.load(CACHE / "emb_items.npy").astype(np.float32)
    q_emb = query_embeddings(queries)

    qi, i = df.qi.to_numpy(), df.i.to_numpy()
    # считаем кусками, чтобы не держать в памяти миллионы векторов сразу
    df["emb"] = np.concatenate([
        np.einsum("ij,ij->i", q_emb[qi[lo:lo + chunk]], emb[i[lo:lo + chunk]])for lo in range(0, len(df), chunk)
    ])
    df.to_parquet(CACHE / f"feats_{mode}.parquet")
    print(f"emb updated: {mode}, {len(df)} rows")


if __name__ == "__main__":
    if sys.argv[1] == "refresh":
        refresh_embeddings(sys.argv[2])
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
