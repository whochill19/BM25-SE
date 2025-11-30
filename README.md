# BM25 Semantic Search

Repo ini berisi implementasi **BM25 + Semantic Search** untuk dokumen.
Project ini menggunakan Python dan dapat dijalankan secara lokal.

---

## **1. Persyaratan**

* Python >= 3.8
* pip
* Virtual environment (opsional tapi disarankan)

---

## **2. Instalasi**

### a. Clone repo

```bash
git clone https://github.com/whochill19/BM25-SE.git
cd BM25-SE
```

### b. Buat virtual environment (opsional tapi disarankan)

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
```

### c. Install dependencies

```bash
pip install -r requirements.txt
```

> **Catatan:** Pastikan `.venv` tidak diikutkan ke Git (sudah di-ignore di `.gitignore`).

---

## **3. Menjalankan project**

1. Pastikan dataset sudah ada di folder `dataset/`.
2. Jalankan script utama:

```bash
python main.py
```

3. Ikuti instruksi di terminal untuk input query dan melihat hasil pencarian.

---

## **4. Struktur folder**

```
BM25-SE/
│
├─ .gitignore
├─ main.py
├─ bm25_model.py
├─ embedding_model.py
├─ preprocess.py
├─ dataset/
│   └─ processed_documents.csv
└─ requirements.txt
```

