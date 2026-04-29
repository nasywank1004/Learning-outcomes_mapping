import pandas as pd
import numpy as np
import os

def normalize(skill):
    """Membersihkan dan menstandarisasi teks skill."""
    if pd.isna(skill) or not isinstance(skill, str):
        return ''
    # Kecilkan huruf, hapus spasi ujung, ganti dash/underscore dengan spasi
    normalized = skill.strip().lower().replace('-', ' ').replace('_', ' ')
    while '  ' in normalized:
        normalized = normalized.replace('  ', ' ')
    return normalized

def get_skill_set(column_value):
    """Mengubah string berisi newline menjadi set skill yang sudah dinormalisasi."""
    if pd.isna(column_value):
        return set()
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

    use_index_as_id = id_col_name not in df_annotator1.columns
    num_rows = min(len(df_annotator1), len(df_annotator2))

    for i in range(num_rows):
        row1 = df_annotator1.iloc[i]
        row2 = df_annotator2.iloc[i]

        system_skills = get_skill_set(row1.get('Hasil Ekstraksi Sistem', ''))
        annotator1_skills = get_skill_set(row1.get('Hasil Anotasi Pakar', ''))
        annotator2_skills = get_skill_set(row2.get('Hasil Anotasi Pakar', ''))

        ground_truth = annotator1_skills.union(annotator2_skills)

        tp = len(system_skills.intersection(ground_truth))
        fp = len(system_skills - ground_truth)
        fn = len(ground_truth - system_skills)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        identifier = row1[id_col_name] if not use_index_as_id else i + 1
            
        results_data.append({
            id_col_name: identifier,
            'Ground Truth': ', '.join(sorted(list(ground_truth))),
            'System Skills': ', '.join(sorted(list(system_skills))),
            'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'F1-Score': f1_score
        })

    results_df = pd.DataFrame(results_data)
    
    # Macro Average untuk print log
    avg_f = results_df['F1-Score'].mean()
    print(f"   - {sheet_name} selesai. Macro F1: {avg_f:.4f}")
    
    return results_df

# --- EKSEKUSI UTAMA ---

folders = ['SkillNER', 'RAKE', 'YAKE', 'KeyBERT']
file_a_name = 'Anotasi Skill - A.xlsx'
file_b_name = 'Anotasi Skill - B.xlsx'

for folder in folders:
    path_a = os.path.join(folder, file_a_name)
    path_b = os.path.join(folder, file_b_name)

    print(f"\n[MEMPROSES FOLDER: {folder.upper()}]")

    if os.path.exists(path_a) and os.path.exists(path_b):
        try:
            # 1. Load Data
            anot1_clo = pd.read_excel(path_a, sheet_name='CLO')
            anot2_clo = pd.read_excel(path_b, sheet_name='CLO')
            anot1_sfia = pd.read_excel(path_a, sheet_name='SFIA')
            anot2_sfia = pd.read_excel(path_b, sheet_name='SFIA')

            # 2. Proses Evaluasi
            df_res_clo = process_and_evaluate(anot1_clo, anot2_clo, 'CLO')
            df_res_sfia = process_and_evaluate(anot1_sfia, anot2_sfia, 'SFIA')

            # 3. Simpan ke file Excel spesifik di dalam folder tersebut
            output_path = os.path.join(folder, f'Evaluasi_{folder}_Final.xlsx')
            with pd.ExcelWriter(output_path) as writer:
                df_res_clo.to_excel(writer, sheet_name='Hasil CLO', index=False)
                df_res_sfia.to_excel(writer, sheet_name='Hasil SFIA', index=False)
            
            print(f"   [SUKSES] File disimpan ke: {output_path}")

        except Exception as e:
            print(f"   [ERROR] Gagal memproses data: {e}")
    else:
        print(f"   [SKIP] File {file_a_name} atau {file_b_name} tidak ditemukan di folder ini.")

print("\n" + "="*30)
print("PROSES SELESAI")