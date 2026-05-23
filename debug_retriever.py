from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)

search_query = "Αν ο ιδιοκτήτης που επέκτεινε την οικοδομή του στο ξένο γήπεδο το έκανε κακόπιστα (γνωρίζοντας ότι το οικόπεδο δεν είναι δικό του), υποχρεούται ο γείτονας να ανεχθεί την οικοδομή έναντι αποζημίωσης"
semantic_docs = db.similarity_search(search_query, k=3)
all_docs = list(db.docstore._dict.values())
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 3
bm25_docs = bm25_retriever.invoke(search_query)

    # 4. Συνδυασμός & Αφαίρεση Διπλοτύπων
    # Ενώνουμε τις δύο λίστες και κρατάμε κάθε άρθρο μόνο μία φορά
final_docs = []
seen_content = set()

# Πρώτα βάζουμε τα αποτελέσματα του BM25 (Exact Matches)
for doc in bm25_docs:
    if doc.page_content not in seen_content:
        final_docs.append(doc)
        seen_content.add(doc.page_content)

# Μετά συμπληρώνουμε με τα Semantic (για να πιάσουμε το "νόημα")
for doc in semantic_docs:
    if doc.page_content not in seen_content:
        final_docs.append(doc)
        seen_content.add(doc.page_content)

print(f"\n🔍 Υβριδικά Αποτελέσματα για: {search_query}")
for i, doc in enumerate(final_docs[:3]): # Κρατάμε τα 3 καλύτερα
    print(f"\n--- Top Result {i+1} ---")
    print(doc.page_content[:300] + "...")