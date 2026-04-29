import pandas as pd
import numpy as np
import os

def normalize(skill):
    """Membersihkan dan menstandarisasi teks skill."""
    if pd.isna(skill) or not isinstance(skill, str):
        return ''
    # Kecilkan huruf, hapus spasi ujung, ganti dash/underscore dengan spasi
    normalized = skill.strip().lower().replace('-', ' ').replace('_', ' ')
    # Hapus double spasi jika ada
    while '  ' in normalized:
        normalized = normalized.replace('  ', ' ')
    return normalized

def get_skill_set(column_value):
    """Mengubah string berisi newline menjadi set skill yang sudah dinormalisasi."""
    if pd.isna(column_value):
        return set()
    # Pisahkan berdasarkan baris baru, bersihkan, dan masukkan ke set
    return {normalize(s) for s in str(column_value).split('\n') if s.strip()}

def process_and_evaluate(df_annotator1, df_annotator2, sheet_name):
    results_data = []

    # Tentukan nama kolom ID berdasarkan sheet
    if sheet_name == 'CLO':
        id_col_name = 'Nama Mata Kuliah'
    elif sheet_name == 'SFIA':
        id_col_name = 'Skill - Level'
    else:
        id_col_name = 'ID'

    # Cek apakah kolom ID ada, jika tidak pakai index
    use_index_as_id = id_col_name not in df_annotator1.columns
    if use_index_as_id:
        print(f"Catatan: Kolom '{id_col_name}' tidak ditemukan di sheet {sheet_name}. Menggunakan index.")

    # Loop berdasarkan jumlah baris terkecil di antara kedua dataframe
    num_rows = min(len(df_annotator1), len(df_annotator2))

    for i in range(num_rows):
        row1 = df_annotator1.iloc[i]
        row2 = df_annotator2.iloc[i]

        # Ambil set skill
        # .get() digunakan agar tidak error jika nama kolom salah/tidak ada
        system_skills = get_skill_set(row1.get('Hasil Ekstraksi Sistem', ''))
        annotator1_skills = get_skill_set(row1.get('Hasil Anotasi Pakar', ''))
        annotator2_skills = get_skill_set(row2.get('Hasil Anotasi Pakar', ''))

        # Ground Truth (Union dari kedua pakar)
        ground_truth = annotator1_skills.union(annotator2_skills)

        # Hitung True Positives, False Positives, False Negatives
        tp = len(system_skills.intersection(ground_truth))
        fp = len(system_skills - ground_truth)
        fn = len(ground_truth - system_skills)
        
        # Hitung Precision, Recall, dan F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Ambil nilai ID
        identifier = row1[id_col_name] if not use_index_as_id else i + 1
            
        # Simpan hasil
        results_data.append({
            id_col_name: identifier,
            'Ground Truth': ', '.join(sorted(list(ground_truth))),
            'System Skills': ', '.join(sorted(list(system_skills))),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })

    # Buat DataFrame hasil
    results_df = pd.DataFrame(results_data)
    
    # Cetak hasil ke konsol
    print(f"\n" + "="*50)
    print(f"HASIL EVALUASI SHEET: {sheet_name}")
    print("="*50)
    
    # Macro Average
    avg_p = results_df['Precision'].mean()
    avg_r = results_df['Recall'].mean()
    avg_f = results_df['F1-Score'].mean()

    print(f"Macro Avg Precision : {avg_p:.4f}")
    print(f"Macro Avg Recall    : {avg_r:.4f}")
    print(f"Macro Avg F1-Score  : {avg_f:.4f}")
    
    return results_df

# --- EKSEKUSI UTAMA ---

file_a = 'Anotasi Skill - A.xlsx'
file_b = 'Anotasi Skill - B.xlsx'

if os.path.exists(file_a) and os.path.exists(file_b):
    try:
        # Load Data
        anot1_clo = pd.read_excel(file_a, sheet_name='CLO')
        anot2_clo = pd.read_excel(file_b, sheet_name='CLO')
        
        anot1_sfia = pd.read_excel(file_a, sheet_name='SFIA')
        anot2_sfia = pd.read_excel(file_b, sheet_name='SFIA')

        # Proses
        results_clo_df = process_and_evaluate(anot1_clo, anot2_clo, 'CLO')
        results_sfia_df = process_and_evaluate(anot1_sfia, anot2_sfia, 'SFIA')

        # Simpan ke Excel
        output_filename = 'Evaluasi_Anotasi_Final.xlsx'
        with pd.ExcelWriter(output_filename) as writer:
            results_clo_df.to_excel(writer, sheet_name='Hasil CLO', index=False)
            results_sfia_df.to_excel(writer, sheet_name='Hasil SFIA', index=False)
        
        print(f"\n[SUKSES] Hasil evaluasi telah disimpan di: {output_filename}")

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat memproses data: {e}")
else:
    print("[ERROR] Salah satu file (Anotasi Skill - A.xlsx atau B.xlsx) tidak ditemukan!")