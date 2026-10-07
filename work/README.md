# Week 5 — Geo-Satellites Classification (20 classes, Accuracy)

## Что готово
Полный пайплайн в `work/`:
- `common.py` — сиды, Dataset, train/eval/predict (переиспользуется всеми экспериментами);
- `compare.py` — базовые эксперименты: **MLP**, **простая CNN**, **advanced CNN**, **ResNet-18 (pretrained)** на 64px;
- `check_artifacts.py` — диагностика артефактов val/test (blur/invert/JPEG/разрешение);
- `final_experiment.py` — финальная модель ResNet-18 + robust-augmentation (blur/noise/invert/sharpness), 128px, режимы `train | finetune | scratch`;
- `final_submission.py` — финальный обученный пайплайн: train+val → предсказание test → `final_submission.csv`;
- `predict_test.py` — генерация предсказаний из сохранённой модели (<2 мин после fit);
- `run_all.sh` — воспроизведение всех экспериментов одной командой.

## Как запустить с реальными данными
1. Положить содержимое архива соревнования (`train.csv`, `val.csv`, `test.csv`, папки `train/`, `val/`, `test/`) в любую папку, например `./week5_data`.
2. Из `/workspace/work`:
   ```bash
   DATA_DIR=/path/to/week5_data ./run_all.sh
   ```
   или только финальный шаг:
   ```bash
   DATA_DIR=/path/to/week5_data python3 final_submission.py /path/to/week5_data final_submission.csv 128
   ```
3. Загрузить `final_submission.csv` на Kaggle (формат `id,label`, метка 0–19).

## Статус проверки
Все скрипты протестированы «end-to-end» на синтетическом датасете той же структуры
(`make_synthetic.py`, 20 классов): MLP/CNN обучаются, ResNet-18 (ImageNet-веса, разрешено правилами)
обучается и генерирует корректный submission-файл. Метрики на синтетике не имеют отношения к реальным данным.

## Требования правил, учтённые в коде
- Сравнение MLP / CNN / pretrained (ResNet-18) — есть; трансформеры не используются.
- Только ImageNet-веса torchvision (из разрешённого списка), без внешних данных/эмбеддингов.
- test.csv используется исключительно для инференса.
- Единственный источник сидов — `seed_everything()` (Python/NumPy/Torch/cuDNN-deterministic).
- Финальная модель дообучается на train+val (разрешено командной фазой); held-out метрика — val-accuracy
  экспериментов `final_experiment.py`, где val не участвует в фитtinge.

## Ограничения запуска здесь
Реальный датасет (~300 МБ) в этой среде отсутствует — скачивание с Kaggle невозможно.
Оценка времени на реальных данных: CPU-only, ~4999+699 изображений 128px, 25 эпох ≈ 2–3 часа,
что превышает лимит 45 минут Kaggle-CPU — см. рекомендации ниже.

## Рекомендации для ускорения под лимит Kaggle (45 мин, CPU, ≤16 ГБ RAM)
1. `IMG_SIZE = 96` вместо 128 (проверьте точность на val — обычно −1…2%).
2. В `final_submission.py` уменьшить `epochs=25 → 12–15` (cosine scheduler уже есть).
3. Использовать `model_final.pt`/предобученный чекпойнт только как результат этого же ноутбука (по правилам ноутбук обязан тренировать сам).
4. При необходимости — `unfreeze="layer4"` вместо `"all"` (быстрее почти в 2 раза, потеря точности мала при mixup).
5. `BATCH_SIZE` поднять до 128 если RAM позволяет.
