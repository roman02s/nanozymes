# API service
## Описание

Схема, описывающая логику сервиса: 
https://miro.com/app/board/uXjVNdSWWvM=/?share_link_id=404762759941

Сервис для доступа по API:
1) POST `/nanozymes_bot`
```json
POST /nanozymes_bot
{
	"<article>": { // статья, по которой пользователь спрашивает вопрос
			"doi_url": <doi_url>,
			"text_with_parameters": {...} // параметры для соответствующей статьи
	}
	"query_text": "<текст запроса>",
	"context": "<предыдущие сообщения пользователя>",
}
result:
{
	"answer": "<ответ на запрос с заданным контекстом пользователю>",
	"context": "<предыдущие сообщения пользователя + новый запрос>",
}
```
2) POST `/nanozymes_bot`
```json
POST /find_simulary
{
	"params": {"v_max": <v_max_value>, "K_m": <K_m_value>},
}
result:
{
	"articles": { // возвращем top-5 статей
		"<article_1>": {
			"doi_url": <doi_url>,
			"text_with_parameters": {...} // параметры для соответствующей статьи
		}
	},
}
```

## Команда
- Роман Сим
- Роман Одобеску
- Олег Загорулько
- Сабина Мирзаева
- Рустем Хакимуллин


