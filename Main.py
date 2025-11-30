import tkinter as tk
from tkinter import ttk
import pandas as pd
import difflib

from bm25_model import BM25
from preprocess import preprocess_query
from embedding_model import load_embedding, semantic_search

# ========================
# LOAD DATA + INIT MODEL
# ========================
df = pd.read_csv("./dataset/processed_documents.csv")

bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(df["processed_document"].tolist())

try:
    model, corpus_embeddings = load_embedding("./dataset/models")
    use_semantic = True
except Exception as e:
    print(f"Tidak bisa memuat embedding: {e}")
    model, corpus_embeddings = None, None
    use_semantic = False

# ========================
# SPELLING CORRECTION
# ========================
def correct_spelling(query, vocab, cutoff=0.8):
    words = query.lower().split()
    corrected = []

    for w in words:
        match = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
        corrected.append(match[0] if match else w)

    new_query = " ".join(corrected)
    return new_query if new_query != query else None

# ========================
# GUI APPLICATION
# ========================
class MedicineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Search Engine")
        self.root.geometry("900x600")

        self.df = df

        # ---- Query Input ----
        frame = ttk.Frame(root)
        frame.pack(pady=10)

        ttk.Label(frame, text="Cari Obat / Gejala:").pack(side=tk.LEFT, padx=5)
        self.entry = ttk.Entry(frame, width=50)
        self.entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Search", command=self.search).pack(side=tk.LEFT, padx=5)

        # ---- Output Text ----
        self.output = tk.Text(root, wrap="word", height=35)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

        scroll = ttk.Scrollbar(self.output, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Highlight style
        self.output.tag_config("highlight", background="yellow", foreground="black")

    # ========================
    # Highlight Helper
    # ========================
    def _insert_highlight_text(self, text, terms):
        for word in text.split():
            w = word.lower()
            if any(t in w for t in terms):
                self.output.insert(tk.END, word + " ", "highlight")
            else:
                self.output.insert(tk.END, word + " ")

    # ========================
    # Display Results
    # ========================
    def display_results(self, results, query, method="BM25", corrected_query=None):
        self.output.delete("1.0", tk.END)

        if method == "BM25":
            self.output.insert(tk.END, f"Hasil BM25 untuk: {query}\n")
        else:
            if corrected_query and corrected_query != query:
                self.output.insert(tk.END, f"Hasil Semantic Search untuk query yang diperbaiki: '{corrected_query}' (awal: '{query}')\n")
            else:
                self.output.insert(tk.END, f"Hasil Semantic Search untuk: {query}\n")
        
        self.output.insert(tk.END, f"{len(results)} hasil ditemukan.\n\n")

        terms = corrected_query.lower().split() if corrected_query else query.lower().split()

        for rank, (doc_index, score) in enumerate(results, 1):
            row = self.df.iloc[doc_index]

            name = row.get("Medicine Name", "Tidak tersedia")
            uses = str(row.get("Uses", "Tidak tersedia") or "Tidak tersedia")
            comp = str(row.get("Composition", "Tidak tersedia") or "Tidak tersedia")
            side = str(row.get("Side_effects", "Tidak tersedia") or "Tidak tersedia")

            self.output.insert(tk.END, f"{rank}. {name}  (score: {score:.4f})\n")
            self.output.insert(tk.END, "   Kegunaan     : ")
            self._insert_highlight_text(uses, terms)
            self.output.insert(tk.END, "\n   Komposisi    : ")
            self._insert_highlight_text(comp, terms)
            self.output.insert(tk.END, f"\n   Efek samping : {side}\n\n")

    # ========================
    # Search Button
    # ========================
    def search(self):
        query = self.entry.get().strip()

        if not query:
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Masukkan kata kunci pencarian.\n")
            return

        processed_query = preprocess_query(query)

        # BM25 search
        results_bm25 = bm25.search(processed_query, top_k=10)
        MIN_BM25 = 0.1
        results_bm25 = [(idx, score) for idx, score in results_bm25 if score >= MIN_BM25]

        if results_bm25:
            self.display_results(results_bm25, query, method="BM25")
            return

        # Fallback semantic search
        if use_semantic:
            corpus_vocab = set(" ".join(df["processed_document"]).split())
            corrected = correct_spelling(query, corpus_vocab)
            final_query = corrected if corrected else query

            results_semantic = semantic_search(
                query=final_query,
                model=model,
                corpus_embeddings=corpus_embeddings,
                df=df,
                top_k=5
            )

            if results_semantic:
                self.display_results(results_semantic, query, method="Semantic Search", corrected_query=corrected)
            else:
                self.output.delete("1.0", tk.END)
                self.output.insert(tk.END, f"❌ Tidak ditemukan hasil relevan untuk: {query}\n")
        else:
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "❌ Tidak ada hasil relevan dan semantic search tidak aktif.\n")


# ========================
# RUN APP
# ========================
root = tk.Tk()
app = MedicineGUI(root)
root.mainloop()
