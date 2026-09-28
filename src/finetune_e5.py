# Дообучение e5 на парах (запрос, выбранное объявление) из train, нужен GPU
# InfoNCE, негативы берутся из батча. Группы, которые идут в val, в обучение не попадают
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (AutoModel, 
                          AutoTokenizer, 
                          PreTrainedModel,
                          PreTrainedTokenizerBase)

from config import CACHE, ITEMS, TRAIN, norm
from embed import DEVICE, MODEL, encode, item_text, model
from features import val_queries

# чем больше батч, тем больше негативов у каждого примера
BATCH = 256
EPOCHS = 2
# температура делает распределение внутри батча резче, 0.05 типичное
# значение для e5
TEMPERATURE = 0.05
OUT = CACHE / "e5-ft"

TRAIN_COLUMNS = [
    "search_query", "search_location_id", "search_infm_params_text",
    "item_id", "item_microcat_id", "item_title_raw", "item_infm_params_text",
]


# пары (текст запроса, текст выбранного объявления) из train
def training_pairs() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN, columns=TRAIN_COLUMNS)
    train["q"] = train.search_query.fillna("").map(norm)
    item_ids = set(pd.read_parquet(ITEMS, columns=["item_id"]).item_id)

    # те же группы, что в features.py val 12000
    _, rest = val_queries(train, item_ids, 12000)
    pairs = pd.DataFrame({"query": rest.q, "item": item_text(rest)})

    # дубликаты убираем и перемешиваем, чтобы в батче были разные запросы
    return pairs.drop_duplicates().sample(frac=1, random_state=0)


# то же, что embed.encode: mean pooling и нормировка, но с градиентами
def embed_batch(encoder: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, texts: list[str], max_len: int) -> torch.Tensor:
    
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
    hidden = encoder(**tokens).last_hidden_state
    mask = tokens["attention_mask"].unsqueeze(-1)

    return F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)


# для каждого запроса правильный ответ - его объявление, остальные объявления батча служат негативами. Модель учится ставить правильное выше остальных
def infonce_loss(queries: torch.Tensor, items: torch.Tensor, item_ids: np.ndarray) -> torch.Tensor:
    logits = queries @ items.T / TEMPERATURE

    # если в батче попались одинаковые объявления, не считаем их негативами друг для друга
    ids = torch.tensor(item_ids, device=DEVICE)
    eye = torch.eye(len(ids), dtype=torch.bool, device=DEVICE)
    same = (ids[:, None] == ids[None]) & ~eye
    logits = logits.masked_fill(same, -1e4)

    target = torch.arange(len(ids), device=DEVICE)
    
    return F.cross_entropy(logits, target)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoder = AutoModel.from_pretrained(MODEL).to(DEVICE).train()
    # маленький learning rate, чтобы не испортить то, что модель уже умеет
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-5)
    # обучение в float16 быстрее и занимает меньше видеопамяти, GradScaler не дает маленьким градиентам обнулиться
    scaler = torch.amp.GradScaler()

    pairs = training_pairs()
    steps = len(pairs) // BATCH
    print(f"{len(pairs)} pairs, {steps} steps/epoch")

    for epoch in range(EPOCHS):
        for step in range(steps):
            batch = pairs.iloc[step * BATCH:(step + 1) * BATCH]
            query_texts = ["query: " + t for t in batch["query"]]
            item_texts = ["passage: " + t for t in batch["item"]]

            with torch.autocast(DEVICE, dtype=torch.float16):
                q = embed_batch(encoder, tokenizer, query_texts, 24)
                p = embed_batch(encoder, tokenizer, item_texts, 48)
                loss = infonce_loss(q, p, pd.factorize(batch["item"])[0])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps} "
                      f"loss {loss.item():.3f}", flush=True)

    # сохраняем модель и сразу пересчитываем эмбеддинги корпуса новой моделью
    encoder.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)

    model.cache_clear()  # чтобы encode подхватил новую модель
    columns = ["item_title_raw", "item_infm_params_text"]
    items = pd.read_parquet(ITEMS, columns=columns)
    embeddings = encode(item_text(items).tolist(), "passage: ")
    np.save(CACHE / "emb_items.npy", embeddings)
    print("done")


if __name__ == "__main__":
    main()
