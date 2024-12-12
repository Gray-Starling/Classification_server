# Classification server

## Описание проекта

Этот проект представляет собой сервер для классификации новостей, написанный на Python. Он использует машинное обучение для анализа текстов новостей и их классификации по заданным категориям. Сервер автоматически загружает необходимые данные и обучает модель, если они отсутствуют, а затем предоставляет API для классификации текстов.

## Функционал

- **Проверка наличия модели и датасета:** При запуске скрипта сервер проверяет, существуют ли уже обученная модель и датасет. Если они отсутствуют, сервер автоматически загружает датасет и обучает модель.
- **Обучение модели:** После загрузки датасета сервер обучает модель классификации и сохраняет её для дальнейшего использования.
- **Классификация текстов:** Сервер работает в режиме ожидания POST-запросов. При получении текста новостей он классифицирует его и возвращает предсказание.

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Gray-Starling/Classification_server.git
cd Classification_server
```

2. Установите необходимые зависимости:

```bash
pip install -r requirements.txt
```

## Запуск сервера

Для запуска сервера выполните следующую команду:

```bash
python start.py
```

Сервер будет запущен и готов принимать POST-запросы для классификации текстов. Если модель и датасет не будут найдены, скрипт скачает датасет и обучить модель.

[Сервер](https://github.com/Gray-Starling/news_parser) по сбору датасета

## Использование API

Сервер предоставляет API для классификации новостей. Вы можете отправить POST-запрос с текстом новости в формате JSON:

### Пример запроса

```bash
curl -X POST http://localhost:5000/classify \
-H "Content-Type: application/json" \
-d '{"text": "Ваш текст новости здесь"}'
```

### Пример ответа
```bash
{
  "predict": "политика",
  "status": "predicted"
}
```

## Лицензия

Этот проект лицензирован под MIT License. См. файл LICENSE для получения дополнительной информации.

## Контрибьюция

Если вы хотите внести свой вклад в проект, пожалуйста, создайте форк репозитория, внесите изменения и отправьте пулл-реквест.

## Автор

[Gray-Starling](https://github.com/Gray-Starling)

---

Спасибо за использование нашего сервера для классификации новостей! Если у вас есть вопросы или предложения, не стесняйтесь обращаться.
