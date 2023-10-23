from src.get_context import get_context


def find_similary(document, query_text):
    context = get_context(document, query_text)
    return context

