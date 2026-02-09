import pandas as pd
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.skill_extraction import (
    extract_skills_skillner,
    extract_skills_skillner_qe,
    extract_keybert_keywords,
    extract_rake_keywords,
    extract_yake_keywords
)

# --- FUNGSI FILTERING ---
def create_skill_filter(sfia_df):
    sfia_skill_names = sfia_df['Skill'].unique().tolist()
    vectorizer = TfidfVectorizer()
    sfia_vectors = vectorizer.fit_transform(sfia_skill_names)
    return vectorizer, sfia_vectors

def filter_extracted_skills(skills_to_filter, vectorizer, sfia_vectors, threshold=0.1):
    if not isinstance(skills_to_filter, list) or not skills_to_filter:
        return []
    
    skills_to_filter_str = [str(s) for s in skills_to_filter]
    skill_vectors = vectorizer.transform(skills_to_filter_str)
    
    if skill_vectors.shape[0] == 0:
        return []
        
    similarity_matrix = cosine_similarity(skill_vectors, sfia_vectors)
    filtered_skills = []
    
    for i in range(len(skills_to_filter_str)):
        if similarity_matrix[i].max() >= threshold:
            filtered_skills.append(skills_to_filter_str[i])
            
    return filtered_skills

# --- FUNGSI PEMBANTU ---
def safe_get(df, col_name):
    return df[col_name].fillna('') if col_name in df.columns else pd.Series([''] * len(df))

def count_skills_per_row(skills_column):
    return skills_column.apply(lambda x: len(x) if isinstance(x, list) else 0)

# --- FUNGSI UTAMA ---
def extract_all_skills(course_file: str, sfia_file: str, ps_name: str, filter_threshold=0.1):
    courses_df = pd.read_excel(course_file)
    sfia_df = pd.read_excel(sfia_file)

    # Inisialisasi Filter
    skill_vectorizer, sfia_skill_vectors = create_skill_filter(sfia_df)

    # Pre-processing text
    courses_df['merged_text'] = (
        safe_get(courses_df, 'course_description_cleaned') + ' ' +
        safe_get(courses_df, 'LO_cleaned') + ' ' +
        safe_get(courses_df, 'course_content_cleaned')
    )

    # Daftar model sesuai urutan tabel Anda
    models_mapping = {
        'SkillNER': 'SkillNER',
        'SkillNER + Query Expansion': 'SkillNER_QE',
        'SkillNER + RAKE': 'SkillNER_RAKE',
        'SkillNER + YAKE': 'SkillNER_YAKE',
        'SkillNER + KeyBERT': 'SkillNER_KeyBERT'
    }

    results_summary = []

    # Loop untuk dataset SFIA dan dataset Universitas
    datasets = [
        (sfia_df, 'Level_Description_cleaned', 'SFIA'),
        (courses_df, 'merged_text', f'SI {ps_name.split("_")[0]}') # Contoh: SI UNAIR atau SI ITS
    ]

    for df, text_col, dataset_label in datasets:
        print(f"Processing {dataset_label}...")
        
        # Ekstraksi Dasar
        df['SkillNER'] = df[text_col].apply(extract_skills_skillner)
        df['SkillNER_QE'] = df[text_col].apply(extract_skills_skillner_qe)
        rake = df[text_col].apply(extract_rake_keywords)
        yake = df[text_col].apply(extract_yake_keywords)
        kbert = df[text_col].apply(extract_keybert_keywords)

        # Ensemble
        df['SkillNER_RAKE'] = [list(set(a).union(b)) for a, b in zip(df['SkillNER'], rake)]
        df['SkillNER_YAKE'] = [list(set(a).union(b)) for a, b in zip(df['SkillNER'], yake)]
        df['SkillNER_KeyBERT'] = [list(set(a).union(b)) for a, b in zip(df['SkillNER'], kbert)]

        # Hitung statistik untuk setiap model sesuai urutan gambar
        for display_name, col_name in models_mapping.items():
            # 1. Sebelum Filtering
            count_before = count_skills_per_row(df[col_name]).sum()
            
            # 2. Proses Filtering
            df[col_name] = df[col_name].apply(
                lambda x: filter_extracted_skills(x, skill_vectorizer, sfia_skill_vectors, filter_threshold)
            )
            
            # Khusus SFIA: Tambahkan kembali skill aslinya agar tidak hilang jika similarity rendah
            if dataset_label == 'SFIA':
                 df[col_name] = [list(set(x + [s])) for x, s in zip(df[col_name], df['Skill'])]

            # 3. Sesudah Filtering
            count_after = count_skills_per_row(df[col_name]).sum()

            results_summary.append({
                'Jenis Dataset': dataset_label,
                'Model': display_name,
                'Jumlah Skill Sebelum Filtering': count_before,
                'Jumlah Skill Sesudah Filtering': count_after
            })

    # Simpan hasil tabel summary
    summary_df = pd.DataFrame(results_summary)
    output_dir = f"Output3/{ps_name}/Ekstraksi"
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_excel(f"{output_dir}/tabel_summary_filtering_{ps_name}.xlsx", index=False)
    
    return summary_df

if __name__ == '__main__':
    program_studies = ["UNAIR_IS", "ITS_IS"]
    for ps in program_studies:
        summary = extract_all_skills(
            f"Output3/{ps}/Preprocessing/processed_courses_{ps}.xlsx",
            f"Output3/{ps}/Preprocessing/processed_sfia_{ps}.xlsx",
            ps
        )
        print(f"\n--- Hasil Tabel untuk {ps} ---")
        print(summary.to_string(index=False))