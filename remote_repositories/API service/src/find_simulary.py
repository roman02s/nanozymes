from src.get_context import get_context

# import logging
# logger = logging.getLogger('nanozymes_bot')
# logger.setLevel(logging.INFO)

# # Создаем обработчик для записи логов в файл
# file_handler = logging.FileHandler('logs/nanozymes_bot.log')
# file_handler.setLevel(logging.INFO)

# # Создаем форматтер для записи логов в удобочитаемом формате
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # Добавляем обработчик к логгеру
# logger.addHandler(file_handler)


# logger.info("SUCCESS IN FIND SIMULARY")

def find_simulary(document, query_text):
    # logger.info("RUN IN FIND SIMULARY")
    context = get_context(document, query_text)
    # logger.info(f"context: {context}")
    return context


# logger.info("~SUCCESS IN FIND SIMULARY")
