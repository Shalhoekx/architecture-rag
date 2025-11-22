# 1. Emdedding Модель.

Была выбрана модель all-MiniLM-L6-v2

Ссылка на репо [text](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main)
Размер чанков: 400
Перекрытие чанков: 80

Скрипт для формирования Embedding и векторных индексов: Project/CreateModel.py
Созданные векторы ChromaDb: Project/vectors/

Результат создания скриптом:
[code]
Сформировано 22 документов
Получили 199 чанков
Формирует ембеддинги
Время выполнения: 0.4543 секунд
Формируем векторы
Время выполнения: 74.4669 секунд
[/code]

Emdedding`s формировались с использованием GPU (CUDA).
