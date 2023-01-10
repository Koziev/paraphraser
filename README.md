# Поэтический перефразировщик

Генеративная модель перефразировки коротких текстов: реплик в диалогах, строк стихов.
Она разработана и поддерживается для использования в проектах [чатбота](https://github.com/Koziev/chatbot) и [генеративной поэзии](https://github.com/Koziev/verslibre).

# Датасет

На huggingface выложена публичная версия датасета: [inkoziev/paraphrases](https://huggingface.co/datasets/inkoziev/paraphrases).

# Обучение

Код обучения: [train_paraphraser_with_gpt2doublehead.py](train_paraphraser_with_gpt2doublehead.py).
Используется класс [transformers.GPT2DoubleHeadsModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2DoubleHeadsModel)
с дополнительной классификационной головой. В обучащих данных есть примеры неправильных перефразировок (см. поле "distractors" в сэмплах),
которые используются в данной схеме файнтюна. Кроме того, из обучения исключается исходная фраза-затравка,
чтобы модель не переобучалась на этих данных.

# Готовая модель

Натренированная модель на huggingface: [inkoziev/paraphraser](https://huggingface.co/inkoziev/paraphraser).
Код с примером вызова модели: [run_paraphraser_with_gpt2doublehead.py](run_paraphraser_with_gpt2doublehead.py).




