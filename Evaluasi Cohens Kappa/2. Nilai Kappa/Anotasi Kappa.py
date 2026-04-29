import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def normalize(skill):
    """Menghilangkan spasi ganda, lowercase, dan standarisasi pemisah."""
    if not isinstance(skill, str): return ""
    return " ".join(skill.strip().lower().replace('-', ' ').replace('_', ' ').split())

def calculate_and_save_kappa(file1, file2, sheet_name, writer):
    print(f"--- Memproses Sheet: {sheet_name} ---")

    # Menentukan kolom ID berdasarkan nama sheet
    id_column = 'Nama Mata Kuliah' if sheet_name == 'CLO' else 'Skill - Level'
    
    try:
        df1 = pd.read_excel(file1, sheet_name=sheet_name)
        df2 = pd.read_excel(file2, sheet_name=sheet_name)
    except Exception as e:
        print(f"Gagal membaca sheet '{sheet_name}': {e}")
        return

    on_column = 'Korpus'
    annotation_column = 'Hasil Anotasi Pakar'
    
    # Mengambil nama annotator dari nama file untuk suffix
    annot1_name = file1.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()
    annot2_name = file2.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()

    # Gabungkan data berdasarkan Korpus
    merged_df = pd.merge(
        df1[[id_column, on_column, annotation_column]], 
        df2[[on_column, annotation_column]], 
        on=on_column, 
        suffixes=(f'_{annot1_name}', f'_{annot2_name}')
    ).dropna(subset=[on_column])

    results = []

    def get_normalized_skills(annotation):
        if pd.isna(annotation): return set()
        return {normalize(s) for s in str(annotation).split('\n') if s.strip()}

    for _, row in merged_df.iterrows():
        # 1. Ambil kata-kata dari Korpus sebagai basis 'd' (True Negative)
        corpus_words = {normalize(w) for w in str(row[on_column]).replace(',', ' ').split() if w.strip()}
        
        # 2. Ambil skill yang dianotasi
        anot1_skills = get_normalized_skills(row[f'{annotation_column}_{annot1_name}'])
        anot2_skills = get_normalized_skills(row[f'{annotation_column}_{annot2_name}'])
        
        # 3. Semesta: Gabungan korpus + semua yang dianotasi
        all_possible_items = sorted(list(corpus_words.union(anot1_skills).union(anot2_skills)))
        
        if not all_possible_items:
            results.append({
                id_column: row[id_column], 'a': 0, 'b': 0, 'c': 0, 'd': 0, 
                'po': 1.0, 'pe': 1.0, 'kappa': 1.0
            })
            continue

        # Vektor biner
        y_true = [1 if item in anot1_skills else 0 for item in all_possible_items]
        y_pred = [1 if item in anot2_skills else 0 for item in all_possible_items]

        # Hitung Confusion Matrix: labels=[1, 0] -> a=TP, b=FN, c=FP, d=TN
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        a, b, c, d = cm.ravel() 

        # --- Perhitungan Po dan Pe Manual ---
        total = a + b + c + d
        po = (a + d) / total
        
        # Probabilitas marginal
        p_yes = ((a + b) / total) * ((a + c) / total)
        p_no = ((c + d) / total) * ((b + d) / total)
        pe = p_yes + p_no

        # Hitung Kappa menggunakan sklearn
        kappa = cohen_kappa_score(y_true, y_pred)

        results.append({
            id_column: row[id_column],
            'a': a, 'b': b, 'c': c, 'd': d,
            'po': round(po, 4), 
            'pe': round(pe, 4),
            'kappa': round(kappa, 4)
        })

    # Simpan ke DataFrame dan Excel
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        avg_kappa = results_df['kappa'].mean()
        print(f"Nilai Rata-rata Kappa untuk {sheet_name}: {avg_kappa:.4f}")
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

# --- Main Execution ---
file_anot1 = 'Anotasi Skill - B.xlsx'
file_anot2 = 'Anotasi Skill - C.xlsx'
output_name = f"Kappa_Calculated_{file_anot1.split()[-1].split('.')[0]}_vs_{file_anot2.split()[-1].split('.')[0]}.xlsx"

with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
    calculate_and_save_kappa(file_anot1, file_anot2, 'CLO', writer)
    calculate_and_save_kappa(file_anot1, file_anot2, 'SFIA', writer)

print(f"\nSelesai! File disimpan dengan nama: {output_name}")