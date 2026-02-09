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

# --- FUNGSI FILTERING (TAMBAHAN) ---
def create_skill_filter(sfia_df):
    """Membuat 'kamus' validasi dari NAMA skill asli SFIA."""
    sfia_skill_names = sfia_df['Skill'].unique().tolist()
    vectorizer = TfidfVectorizer()
    sfia_vectors = vectorizer.fit_transform(sfia_skill_names)
    return vectorizer, sfia_vectors

def filter_extracted_skills(skills_to_filter, vectorizer, sfia_vectors, threshold=0.1):
    """Memfilter daftar skill berdasarkan kemiripan dengan kamus SFIA."""
    if not isinstance(skills_to_filter, list) or not skills_to_filter:
        return []
    
    # Pastikan semua elemen adalah string
    skills_to_filter_str = [str(s) for s in skills_to_filter]
    filtered_skills = []
    
    # Transformasi input menjadi vector
    skill_vectors = vectorizer.transform(skills_to_filter_str)
    
    # Hitung similarity
    similarity_matrix = cosine_similarity(skill_vectors, sfia_vectors)
    
    for i in range(len(skills_to_filter_str)):
        # Ambil nilai kemiripan tertinggi untuk setiap kandidat skill
        if similarity_matrix[i].max() >= threshold:
            filtered_skills.append(skills_to_filter_str[i])
            
    return filtered_skills

# --- FUNGSI PEMBANTU (ORIGINAL) ---
def safe_get(df, col_name):
    """Return kolom jika ada, kalau tidak ada return Series kosong"""
    if col_name in df.columns:
        return df[col_name].fillna('')
    else:
        return pd.Series([''] * len(df), index=df.index)

def count_skills_per_row(skills_column):
    return skills_column.apply(lambda x: len(x) if isinstance(x, list) else 0)

# --- FUNGSI UTAMA ---
def extract_all_skills(course_file: str, sfia_file: str, ps_name: str, filter_threshold=0.1):
    courses_df = pd.read_excel(course_file)
    sfia_df = pd.read_excel(sfia_file)

    # Inisialisasi Filter
    print(f"[{ps_name}] Menyiapkan kamus filter SFIA...")
    skill_vectorizer, sfia_skill_vectors = create_skill_filter(sfia_df)

    # === 1. PROSES EKSTRAKSI COURSES ===
    print(f"[{ps_name}] Mengekstraksi keterampilan dari data Courses...")
    courses_df['merged_text'] = (
        safe_get(courses_df, 'course_description_cleaned') + ' ' +
        safe_get(courses_df, 'LO_cleaned') + ' ' +
        safe_get(courses_df, 'course_content_cleaned')
    )

    # Ekstraksi dasar
    courses_df['SkillNER'] = courses_df['merged_text'].apply(extract_skills_skillner)
    courses_df['SkillNER_QE'] = courses_df['merged_text'].apply(extract_skills_skillner_qe)
    courses_df['RAKE'] = courses_df['merged_text'].apply(extract_rake_keywords)
    courses_df['YAKE'] = courses_df['merged_text'].apply(extract_yake_keywords)
    courses_df['KeyBERT'] = courses_df['merged_text'].apply(extract_keybert_keywords)

    # Logika Penggabungan (Ensemble)
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

    # Logika Penggabungan (Ensemble)
    sfia_df['SkillNER_RAKE'] = sfia_df.apply(lambda r: list(set(r['SkillNER']).union(r['RAKE'])), axis=1)
    sfia_df['SkillNER_YAKE'] = sfia_df.apply(lambda r: list(set(r['SkillNER']).union(r['YAKE'])), axis=1)
    sfia_df['SkillNER_KeyBERT'] = sfia_df.apply(lambda r: list(set(r['SkillNER']).union(r['KeyBERT'])), axis=1)

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

    # === 3. FILTERING (TAHAPAN PENYARINGAN) ===
    print(f"[{ps_name}] Menerapkan filtering SFIA Similarity (Threshold={filter_threshold})...")
    for col in skill_columns:
        # Filter untuk Courses
        courses_df[col] = courses_df[col].apply(
            lambda x: filter_extracted_skills(x, skill_vectorizer, sfia_skill_vectors, filter_threshold)
        )
        # Filter untuk SFIA
        sfia_df[col] = sfia_df[col].apply(
            lambda x: filter_extracted_skills(x, skill_vectorizer, sfia_skill_vectors, filter_threshold)
        )

    # === 4. FINALISASI & STATISTIK ===
    # Tambahkan nama skill asli SFIA ke semua list ekstraksi
    for col in skill_columns:
        sfia_df[col] = sfia_df.apply(
            lambda row: list(set(row[col] + [row['Skill']])) if isinstance(row[col], list) else [row['Skill']],
            axis=1
        )

    # Hitung Statistik
    courses_stats, sfia_stats = [], []
    for df, stats_list, label in [(courses_df, courses_stats, "Courses"), (sfia_df, sfia_stats, "SFIA")]:
        print(f"Menghitung statistik untuk {label}...")
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
    output_dir = f"Output2/{ps_name}/Ekstraksi"
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(f"{output_dir}/skill_count_statistics_{ps_name}.xlsx", engine='openpyxl') as writer:
        pd.DataFrame(courses_stats).to_excel(writer, sheet_name="Courses", index=False)
        pd.DataFrame(sfia_stats).to_excel(writer, sheet_name="SFIA", index=False)

    return courses_df, sfia_df


if __name__ == '__main__':
    start_total = time.time()
    
    # Konfigurasi
    program_studies = ["UNAIR_IS", "ITS_IS"]
    THRESHOLD_FILTER = 0.1  # Nilai ambang batas kemiripan SFIA

    for ps in program_studies:
        print(f"\n" + "="*50)
        print(f"PROCESSING DATA: {ps}")
        print("="*50)
        
        # Eksekusi
        c_res, s_res = extract_all_skills(
            f"Output2/{ps}/Preprocessing/processed_courses_{ps}.xlsx",
            f"Output2/{ps}/Preprocessing/processed_sfia_{ps}.xlsx",
            ps,
            filter_threshold=THRESHOLD_FILTER
        )

        # Simpan file hasil ekstraksi
        output_path = f"Output2/{ps}/Ekstraksi"
        os.makedirs(output_path, exist_ok=True)
        
        c_res.to_excel(f"{output_path}/skills_extracted_courses_{ps}.xlsx", index=False)
        s_res.to_excel(f"{output_path}/skills_extracted_sfia_{ps}.xlsx", index=False)
        
        print(f"Files saved for {ps} in {output_path}")

    print(f"\n" + "="*50)
    print(f"TOTAL RUNTIME: {time.time() - start_total:.2f} detik")
    print("="*50)