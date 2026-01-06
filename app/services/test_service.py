from app.services.search_service import search_service

docs = search_service([
    {"axis": "medical", "query": "deep learning for disease diagnosis"}
], top_k_per_query=5)

for d in docs:
    print(d)

