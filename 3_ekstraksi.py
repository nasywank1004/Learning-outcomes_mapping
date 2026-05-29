import pandas as pd
import os
import time
from utils.skill_extraction import (
    extract_skills_skillner,
    extract_skills_skillner_qe,
    extract_keybert_keywords,
    extract_rake_keywords,
    extract_yake_keywords
)

def safe_get(df, col_name):
    """Return kolom jika ada, kalau tidak ada return Series kosong berisi string kosong"""
    if col_name in df.columns:
        return df[col_name].fillna('')
    else:
        return pd.Series([''] * len(df), index=df.index)

def count_skills_per_row(skills_column):
    return skills_column.apply(lambda x: len(x) if isinstance(x, list) else 0)

def extract_all_skills(course_file: str, sfia_file: str, ps_name: str):
    courses_df = pd.read_excel(course_file)
    sfia_df = pd.read_excel(sfia_file)

    # === 1. PROSES EKSTRAKSI COURSES ===
    print(f"[{ps_name}] Mengekstraksi keterampilan dari data Courses...")
    courses_df['merged_text'] = (
        safe_get(courses_df, 'course_description_cleaned') + ' ' +
        safe_get(courses_df, 'LO_cleaned') + ' ' +
        safe_get(courses_df, 'course_content_cleaned')
    )

    # Ekstraksi dasar (langsung dari teks sumber)
    courses_df['SkillNER'] = courses_df['merged_text'].apply(extract_skills_skillner)
    courses_df['SkillNER_QE'] = courses_df['merged_text'].apply(extract_skills_skillner_qe)
    courses_df['RAKE'] = courses_df['merged_text'].apply(extract_rake_keywords)
    courses_df['YAKE'] = courses_df['merged_text'].apply(extract_yake_keywords)
    courses_df['KeyBERT'] = courses_df['merged_text'].apply(extract_keybert_keywords)

    # Logika Penggabungan (Ensemble) Courses
    # Gabungan 2 Metode
    courses_df['SkillNER_RAKE'] = courses_df.apply(lambda r: list(set(r['SkillNER']).union(r['RAKE'])), axis=1)
    courses_df['SkillNER_YAKE'] = courses_df.apply(lambda r: list(set(r['SkillNER']).union(r['YAKE'])), axis=1)
    courses_df['SkillNER_KeyBERT'] = courses_df.apply(lambda r: list(set(r['SkillNER']).union(r['KeyBERT'])), axis=1)

    # === 2. PROSES EKSTRAKSI SFIA ===
    print(f"[{ps_name}] Mengekstraksi keterampilan dari data SFIA...")
    sfia_df['SkillNER'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner)
    sfia_df['SkillNER_QE'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner_qe)
    sfia_df['RAKE'] = sfia_df['Level_Description_cleaned'].apply(extract_rake_keywords)
    sfia_df['YAKE'] = sfia_df['Level_Description_cleaned'].apply(extract_yake_keywords)
    sfia_df['KeyBERT'] = sfia_df['Level_Description_cleaned'].apply(extract_keybert_keywords)

    # Logika Penggabungan (Ensemble) SFIA
    sfia_df['SkillNER_RAKE'] = sfia_df.apply(lambda r: list(set(r['SkillNER']).union(r['RAKE'])), axis=1)
    sfia_df['SkillNER_YAKE'] = sfia_df.apply(lambda r: list(set(r['SkillNER']).union(r['YAKE'])), axis=1)
    sfia_df['SkillNER_KeyBERT'] = sfia_df.apply(lambda r: list(set(r['SkillNER']).union(r['KeyBERT'])), axis=1)

    # === 3. FINALISASI & STATISTIK ===
    skill_columns = [
        'SkillNER', 
        'SkillNER_QE', 
        'RAKE', 
        'YAKE', 
        'KeyBERT',
        'SkillNER_RAKE', 
        'SkillNER_YAKE', 
        'SkillNER_KeyBERT'
    ]

    # Tambahkan nama skill asli SFIA ke semua list ekstraksi
    for col in skill_columns:
        sfia_df[col] = sfia_df.apply(
            lambda row: list(set(row[col] + [row['Skill']])) if isinstance(row[col], list) else [row['Skill']],
            axis=1
        )

    # Hitung Statistik
    courses_stats, sfia_stats = [], []
    for df, stats_list, label in [(courses_df, courses_stats, "Courses"), (sfia_df, sfia_stats, "SFIA")]:
        print(f"\nMenghitung statistik untuk {label}...")
        for col in skill_columns:
            count_col = col + '_count'
            df[count_col] = count_skills_per_row(df[col])
            stats_list.append({
                'Method': col,
                'Mean': df[count_col].mean(),
                'Min': df[count_col].min(),
                'Max': df[count_col].max(),
                'Non-zero Count': (df[count_col] > 0).sum()
            })

    # Simpan Statistik
    output_dir = f"Output/{ps_name}/Ekstraksi"
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(f"{output_dir}/skill_count_statistics_{ps_name}.xlsx", engine='openpyxl') as writer:
        pd.DataFrame(courses_stats).to_excel(writer, sheet_name="Courses", index=False)
        pd.DataFrame(sfia_stats).to_excel(writer, sheet_name="SFIA", index=False)

    return courses_df, sfia_df


if __name__ == '__main__':
    start_total = time.time()
    program_studies = ["UNAIR_IS", "ITS_IS"]

    for ps in program_studies:
        print(f"\n" + "="*50)
        print(f"PROCESSING DATA: {ps}")
        print("="*50)
        
        c_res, s_res = extract_all_skills(
            f"Output/{ps}/Preprocessing/processed_courses_{ps}.xlsx",
            f"Output/{ps}/Preprocessing/processed_sfia_{ps}.xlsx",
            ps
        )

        c_res.to_excel(f"Output/{ps}/Ekstraksi/skills_extracted_courses_{ps}.xlsx", index=False)
        s_res.to_excel(f"Output/{ps}/Ekstraksi/skills_extracted_sfia_{ps}.xlsx", index=False)
        print(f"Files saved for {ps}")

    print(f"\n" + "="*50)
    print(f"TOTAL RUNTIME: {time.time() - start_total:.2f} detik")
    print("="*50)