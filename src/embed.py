# Эмбеддинги e5. Если в cache/e5-ft лежит дообученная модель, берем ее python src/embed.py -> cache/emb_items.npy
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModel, 
                        AutoTokenizer, 
                        PreTrainedModel,
                        PreTrainedTokenizerBase)

from config import CACHE, ITEMS

# маленькая многоязычная модель, нормально понимает русский
MODEL = "intfloat/multilingual-e5-small"
FINETUNED = CACHE / "e5-ft"
# если есть видеокарта с CUDA, считаем на ней, иначе на процессоре
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def item_text(items: pd.DataFrame) -> pd.Series:
    # текст объявления для модели: заголовок и начало параметров,
    # первых 120 символов хватает на вид и тип услуги
    params = items.item_infm_params_text.fillna("").str[:120]

    return items.item_title_raw.fillna("") + ". " + params


# lru_cache загружает модель один раз, дальше она берется из памяти
@lru_cache
def model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    torch.set_num_threads(8)
    name = FINETUNED if FINETUNED.exists() else MODEL
    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name).to(DEVICE).eval()

    return tokenizer, encoder


# prefix для e5 обязателен: "query: " у запросов и "passage: " у объявлений,
# без него качество заметно хуже
@torch.inference_mode()
def encode(texts: list[str], prefix: str,
           batch_size: int = 256, max_len: int = 48) -> np.ndarray:
    tokenizer, encoder = model()

    # сортируем по длине, так в батчах меньше паддинга
    order = np.argsort([len(t) for t in texts])
    chunks = []

    for lo in range(0, len(texts), batch_size):
        batch = [prefix + texts[j] for j in order[lo:lo + batch_size]]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt")
        tokens = tokens.to(DEVICE)
        hidden = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        vectors = (hidden * mask).sum(1) / mask.sum(1)  # mean pooling
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        # float16 экономит половину памяти, для косинуса точности хватает
        chunks.append(vectors.cpu().numpy().astype(np.float16))

        if lo % (50 * batch_size) == 0:
            print(f"{lo}/{len(texts)}", flush=True)

    # возвращаем векторы в исходный порядок текстов
    result = np.empty((len(texts), chunks[0].shape[1]), np.float16)
    result[order] = np.concatenate(chunks)
    
    return result


def main() -> None:
    columns = ["item_title_raw", "item_infm_params_text"]
    items = pd.read_parquet(ITEMS, columns=columns)
    embeddings = encode(item_text(items).tolist(), "passage: ")
    np.save(CACHE / "emb_items.npy", embeddings)
    print("done")


if __name__ == "__main__":
    main()
