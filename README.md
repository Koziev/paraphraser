# Поэтический перефразировщик

Генеративная модель перефразировки **коротких** текстов: реплик в диалогах, строк стихов.
Она разработана и поддерживается для использования в проектах [чатбота](https://github.com/Koziev/chatbot) и [генеративной поэзии](https://github.com/Koziev/verslibre).

# Датасет

На huggingface.co выложена публичная версия датасета: [inkoziev/paraphrases](https://huggingface.co/datasets/inkoziev/paraphrases).

Вы можете использовать этот датасет для обучения своих моделей, про необходимости
дополняя его другими открытыми русскоязычными данными по перефразировкам, например из датасета [cointegrated/ru-paraphrase-NMT-Leipzig](https://huggingface.co/datasets/cointegrated/ru-paraphrase-NMT-Leipzig).

# Обучение

Код обучения: [train_paraphraser_with_gpt2doublehead.py](train_paraphraser_with_gpt2doublehead.py). В нем используется класс [transformers.GPT2DoubleHeadsModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2DoubleHeadsModel)
с дополнительной классификационной головой. В обучащих данных есть примеры неправильных перефразировок (см. поле "distractors" в сэмплах),
которые используются в данной схеме файнтюна. Кроме того, из обучения исключается исходная фраза-затравка,
чтобы модель не переобучалась на этих данных.

# Готовая модель

Натренированная модель на huggingface: [inkoziev/paraphraser](https://huggingface.co/inkoziev/paraphraser).

Код с примером вызова модели: [run_paraphraser_with_gpt2doublehead.py](run_paraphraser_with_gpt2doublehead.py).

# Сопряженные проекты

Модель для оценки близости двух коротких текстов: [inkoziev/sbert_synonymy](https://huggingface.co/inkoziev/sbert_synonymy)

Репозиторий с кодом тренировки модели: [paraphrase_reranker](https://github.com/Koziev/paraphrase_reranker).
