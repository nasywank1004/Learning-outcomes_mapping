import pandas as pd
import time
import os

# --- Ground Truth ---
def create_ground_truth(gt_path: str, sheet_name: str) -> set:
    gt_df = pd.read_excel(gt_path, sheet_name=sheet_name)
    gt_df.columns = gt_df.columns.str.strip().str.replace('\n', '', regex=True)
    level_columns = [col for col in gt_df.columns if col.startswith('Level')]

    ground_truth_set = set()
    for _, row in gt_df.iterrows():
        skill_name = row['Skill']
        for col in level_columns:
            if row[col] == 1.0:
                level_number = col.split()[1]
                ground_truth_set.add(f"{skill_name} {level_number}")
    return ground_truth_set

# --- Evaluasi satu model ---
def evaluate_single_mapping(mapping_file: str, gt_set: set,
                            model_name: str, ps_name: str, expanded: bool = False) -> pd.DataFrame:
    col_name = 'Matched_SFIA'
    if not os.path.exists(mapping_file):
        print(f"File tidak ditemukan: {mapping_file}")
        return None

    df = pd.read_excel(mapping_file)
    if col_name not in df.columns:
        print(f"Kolom '{col_name}' tidak ditemukan di {mapping_file}")
        return None

    predicted_set = set(df[col_name].dropna().unique())

    # Hitung metrik
    tp = len(predicted_set & gt_set)
    fp = len(predicted_set - gt_set)
    fn = len(gt_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    # Konversi ke persen (%)
    precision *= 100
    recall *= 100
    f1_score *= 100
    accuracy *= 100

    return pd.DataFrame([{
        'Program_Study': ps_name,
        'Model': model_name,
        'Expanded': expanded,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision (%)': round(precision, 2),
        'Recall (%)': round(recall, 2),
        'F1_score (%)': round(f1_score, 2),
        'Accuracy (%)': round(accuracy, 2)
    }])

# --- MAIN ---
if __name__ == '__main__':
    start = time.time()
    program_studies = ["UNAIR_IS", "ITS_IS"]

    MODELS = [
        'SkillNER', 
        'SkillNER_QE',
        'SkillNER_RAKE', 
        'SkillNER_YAKE', 
        'SkillNER_KeyBERT'
    ]

    for ps in program_studies:
        gt_file = f"Data/GT_{ps}.xlsx"
        gt_sheet = "Pemetaan"   # sesuaikan dengan isi file GT
        gt_set = create_ground_truth(gt_file, gt_sheet)

        ps_results = []  # simpan hasil per prodi

        for model in MODELS:
            sim_type = "cosine"

            original_path = f"Tanpa Filtering/{ps}/Mapping/mapping_{sim_type}_{model}_{ps}.xlsx"
            expanded_path = f"Tanpa Filtering/{ps}/Mapping/expanded_mapping_{sim_type}_{model}_{ps}.xlsx"

            res_orig = evaluate_single_mapping(original_path, gt_set, model, ps, expanded=False)
            res_expd = evaluate_single_mapping(expanded_path, gt_set, model, ps, expanded=True)

            if res_orig is not None:
                ps_results.append(res_orig)
            if res_expd is not None:
                ps_results.append(res_expd)

        # simpan hasil khusus prodi ini
        if ps_results:
            final_df = pd.concat(ps_results, ignore_index=True)
            out_path = f"Tanpa Filtering/{ps}/Evaluasi/evaluation_results_{ps}.xlsx"
            final_df.to_excel(out_path, index=False)
            print(f"[{ps}] Hasil evaluasi disimpan ke: {out_path}")
            print(final_df)
        else:
            print(f"[{ps}] Tidak ada hasil evaluasi yang valid.")

    print(f"\nTotal waktu evaluasi: {time.time() - start:.2f} detik")
