from .get_context import get_context

# document = "C4RA15675G.pdf"
# query_text = "query:  Fe3O4 NPs"
# result = get_context(document, query_text)
# print("type(result): ", type(result))
print("SUCCESS IN FIND SIMULARY")

def find_simulary(document, query_text):
    context = get_context(document, query_text)
    return context