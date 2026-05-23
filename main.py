import sys
import io

# Επιβολή UTF-8 για την επικοινωνία με το τερματικό
if sys.stdin.encoding != 'utf-8':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Μετά συνεχίζεις με τα υπόλοιπα imports...

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_community.retrievers import BM25Retriever

from typing import TypedDict
import os
import unicodedata
import re

ollama_url = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

llm = ChatOllama(
    model="llama3.2", 
    temperature=0,
    base_url=ollama_url  
)
# State & Logic
class AgentState(TypedDict):
    query: str
    context: str
    answer: str
    reflection: str
    count: int

# Αρχικοποίηση μοντέλων
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")


def rewrite_query_node(state: AgentState):
    prompt = (
        "SYSTEM: Είσαι εξειδικευμένος βοηθός αναζήτησης για τον Ελληνικό Αστικό Κώδικα.\n"
        "TASK: Μετέτρεψε την ερώτηση του χρήστη σε 3-4 λέξεις-κλειδιά στα ΕΛΛΗΝΙΚΑ για αναζήτηση σε βάση δεδομένων.\n"
        "STRICT RULES:\n"
        "1. Απάντησε ΜΟΝΟ με τις λέξεις-κλειδιά.\n"
        "2. ΜΗΝ χρησιμοποιείς εισαγωγικά, ΜΗΝ γράφεις 'Απάντηση:' ή 'Λέξεις-κλειδιά:'.\n"
        "3. Αν η ερώτηση περιέχει αριθμό άρθρου (π.χ. 1010), συμπεριέλαβε οπωσδήποτε τη λέξη 'ακ' και τον αριθμό.\n"
        f"USER QUESTION: {state['query']}\n"
        "KEYWORDS:"
    )
    res = llm.invoke(prompt)
    
    print(f"🎯 Optimized Search Keywords: {res.content}")
    return {"query": res.content} 

def retrieve_node(state: AgentState):
    query = str(state.get('query', "")).lower()
    
    # 1. Καθαρισμός & Normalization
    query = ''.join(c for c in unicodedata.normalize('NFD', query) if unicodedata.category(c) != 'Mn')
    match = re.search(r'\d+', query)
    search_query = f"ακ {match.group()}" if match else query

    print(f"🔍 Executing Manual Hybrid Search for: '{search_query}'")

    # 2. Φόρτωση FAISS (Semantic)
    db = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
    semantic_docs = db.similarity_search(search_query, k=2)

    # 3. Εκτέλεση BM25 (Keyword)
    all_docs = list(db.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 2
    bm25_docs = bm25_retriever.invoke(search_query)
    
    combined_docs = semantic_docs + bm25_docs
    unique_contents = set()
    final_docs = []
    
    for doc in combined_docs:
        if doc.page_content not in unique_contents:
            final_docs.append(doc)
            unique_contents.add(doc.page_content)

    print(f"✅ Hybrid search found {len(final_docs)} unique context chunks.")
    
    context = "\n".join([d.page_content for d in final_docs])
    return {"context": context}


def generate_node(state: AgentState):
    prompt = (
        "STRICT INSTRUCTION: Answer ONLY in GREEK. Do not use any other language.\n"
        "Είσαι έγκριτος νομικός σύμβουλος ειδικευμένος στον Αστικό Κώδικα.\n"
        "Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ το παρακάτω πλαίσιο για να απαντήσεις.\n"
        "Αν η πληροφορία δεν υπάρχει στο πλαίσιο, πες 'Δεν βρέθηκε ακριβής αναφορά στα άρθρα που ανακτήθηκαν'.\n\n"
        f"Πλαίσιο (Context): {state['context']}\n"
        f"Ερώτηση: {state['query']}\n"
        f"Παρατηρήσεις: {state.get('reflection', '')}\n\n"
        "Απάντηση στα Ελληνικά:"
    )
    res = llm.invoke(prompt)
    return {"answer": res.content, "count": state['count'] + 1}

def reflect_node(state: AgentState):
    prompt = (
        "STRICT INSTRUCTION: Answer ONLY in GREEK. Do not use English, French, or Spanish.\n"
        f"Είσαι αυστηρός κριτής νομικών κειμένων. Εξέτασε την απάντηση: {state['answer']}\n"
        "Αν η απάντηση είναι σωστή και βασίζεται σε άρθρα, πες μόνο 'OK'.\n"
        "Αν η απάντηση λέει ότι δεν βρήκε πληροφορίες, δώσε οδηγίες για καλύτερη αναζήτηση στα Ελληνικά."
    )
    res = llm.invoke(prompt)
    return {"reflection": res.content}

def should_continue(state: AgentState):
    
    if "OK" in state["reflection"] or state["count"] >= 3:
        return END
    return "reflect"

# Στήσιμο Graph

workflow = StateGraph(AgentState)
#Nodes
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflect_node)
#Entry
workflow.set_entry_point("rewrite")
#Entry
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", should_continue)
workflow.add_edge("reflect", "generate")
#Build
app = workflow.compile()

if __name__ == "__main__":
    print("--- Legal AI Agent (Αστικός Κώδικας) ---")
    print("Πληκτρολογήστε 'exit' για να βγείτε.")
    
    while True:
        # Λήψη εισόδου από το τερματικό
        user_query = input("\n🔹 Ερώτηση: ")
        
        if user_query.lower() in ['exit', 'quit', 'έξοδος']:
            print("Τερματισμός...")
            break
            
        # Εκτέλεση του Agentic Workflow 
        print("⏳ Ο Agent επεξεργάζεται τη νομική απάντηση...")
        inputs = {"query": user_query, "count": 0, "reflection": ""}
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    print(f"\n🤖 Απάντηση: {value['answer']}")
                elif key == "reflect":
                    print(f"🧐 Σκέψη/Διόρθωση: {value['reflection']}")