import pandas as pd
import re

def preprocess_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Hapus artefak Excel
    text = re.sub(r"_x000D_[\n\r]*", " ", text)
    
    # 2. HAPUS NOMOR LIST (contoh: "1. ", "2) ", "1.1. ")
    # Pola: Menghapus angka yang diikuti titik atau kurung di awal baris/setelah spasi
    text = re.sub(r'(^|\s)\d+[\.\)]\s?', ' ', text)
    
    # 3. HAPUS BULLET LIST (contoh: "•", "-", "*")
    text = re.sub(r'[•\-\*]\s?', ' ', text)
    
    # 4. Case Folding
    text = text.lower()
    
    # 5. Hapus karakter spesial yang tersisa (kecuali huruf dan angka)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 6. Hapus Whitespace berlebih
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()