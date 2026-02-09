import pandas as pd
import ast
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- SAFE PARSER ---
def safe_to_list(x):
    """Convert cell content into a list of strings."""
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str) and x.strip():
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [str(i) for i in parsed]
            return [x]
        except:
            return [x]
    return []


def expand_skill_levels(mapping_results, sfia):
    """Ekspansi SFIA skill level dengan menambahkan level-level lebih rendah.
       Nilai Similarity diwariskan dari level tertinggi yang sudah terhitung."""

    missing_levels = []

    # loop setiap skill unik di hasil mapping
    for skill in mapping_results['Matched_SFIA'].str.rsplit(" ", 1).str[0].unique():
        # ambil semua level SFIA yang tersedia untuk skill tersebut
        sfia_levels = (
            sfia[sfia['SFIA_Skill_Level'].str.startswith(skill + " ")]
            ['SFIA_Skill_Level']
            .str.rsplit(" ", 1).str[1].astype(int).tolist()
        )

        # cek level yang sudah ada di hasil mapping
        mapped_rows = mapping_results[mapping_results['Matched_SFIA'].str.startswith(skill + " ")]
        mapped_levels = mapped_rows['Matched_SFIA'].str.rsplit(" ", 1).str[1].astype(int).tolist()

        # tambahkan level lebih rendah
        for level, sim, course in zip(
            mapped_rows['Matched_SFIA'].str.rsplit(" ", 1).str[1].astype(int),
            mapped_rows['Similarity'],
            mapped_rows['Course']
        ):
            lower_levels = [l for l in sfia_levels if l < level]
            for lower_level in lower_levels:
                missing_levels.append({
                    "Course": course,
                    "Matched_SFIA": f"{skill} {lower_level}",
                    "Similarity": sim  # pakai similarity level asal
                })

    if missing_levels:
        missing_df = pd.DataFrame(missing_levels)
        return (
            pd.concat([mapping_results, missing_df], ignore_index=True)
            .drop_duplicates(subset=["Course", "Matched_SFIA"])
            .sort_values(by=["Course", "Matched_SFIA"])
        )
    else:
        return mapping_results


# --- MAIN MAPPING ---
def map_courses_to_sfia(courses_file: str, sfia_file: str, model: str, ps_name: str, threshold=0.2):
    courses_df = pd.read_excel(courses_file)
    sfia_df = pd.read_excel(sfia_file)

    # Buat corpus
    course_corpus = courses_df[model].apply(lambda x: " ".join(safe_to_list(x)))
    sfia_corpus = sfia_df[model].apply(lambda x: " ".join(safe_to_list(x)))

    docs = pd.concat([course_corpus, sfia_corpus])
    docs = docs[docs.str.strip() != ""]

    if docs.empty:
        raise ValueError(f"[{ps_name} - {model}] No valid text found for TF-IDF.")

    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs)

    all_matches = []
    visualized = False

    for _, course_row in courses_df.iterrows():
        course_skills = safe_to_list(course_row[model])
        if not course_skills:
            continue

        course_vector = vectorizer.transform([" ".join(course_skills)])
        feature_names = vectorizer.get_feature_names_out()

        for _, sfia_row in sfia_df.iterrows():
            sfia_skills = safe_to_list(sfia_row[model])
            if not sfia_skills:
                continue

            sfia_vector = vectorizer.transform([" ".join(sfia_skills)])
            score = cosine_similarity(course_vector, sfia_vector)[0][0]

            # Filtered (≥threshold)
            if score >= threshold:
                all_matches.append({
                    "Course": course_row['Course Title'],
                    "Matched_SFIA": sfia_row['SFIA_Skill_Level'],
                    "Similarity": score
                })

            # Simpan 1 contoh TF-IDF comparison
            if not visualized:
                visualized = True
                course_tf = course_vector.toarray().flatten()
                sfia_tf = sfia_vector.toarray().flatten()

                df_compare = pd.DataFrame({
                    "Term": feature_names,
                    "TF-IDF Course": course_tf,
                    "TF-IDF SFIA": sfia_tf
                })
                df_compare = df_compare[(df_compare["TF-IDF Course"] > 0) | (df_compare["TF-IDF SFIA"] > 0)]
                df_compare["Course Title"] = course_row["Course Title"]
                df_compare["SFIA Skill Level"] = sfia_row["SFIA_Skill_Level"]

                metadata = pd.DataFrame({
                    "Keterangan": ["Course Title", "SFIA Skill Level", "Cosine Similarity"],
                    "Nilai": [course_row["Course Title"], sfia_row["SFIA_Skill_Level"], score]
                })

                os.makedirs(f"Output3/{ps_name}/Mapping", exist_ok=True)
                with pd.ExcelWriter(f"Output3/{ps_name}/Mapping/contoh_vektor_perbandingan_{model}.xlsx") as writer:
                    df_compare.to_excel(writer, index=False, sheet_name="TF-IDF Comparison")
                    metadata.to_excel(writer, index=False, sheet_name="Metadata")

    # Buat dataframe hasil (filtered only)
    if not all_matches:
        print(f"Tidak ada hasil mapping untuk {ps_name} - {model}")
        return

    filtered_df = pd.DataFrame(all_matches).sort_values(by=["Course", "Similarity"], ascending=[True, False])
    filtered_df.to_excel(f"Output3/{ps_name}/Mapping/mapping_cosine_{model}_{ps_name}.xlsx", index=False)

    # Ekspansi SFIA skill level pakai expand_skill_levels
    expanded_df = expand_skill_levels(filtered_df, sfia_df)
    expanded_df.to_excel(f"Output3/{ps_name}/Mapping/expanded_mapping_cosine_{model}_{ps_name}.xlsx", index=False)


    print(f"[{ps_name}] Mapping '{model}' selesai. ({len(filtered_df)} baris hasil awal).")
    print(f"Unique sebelum ekspansi: {filtered_df['Matched_SFIA'].nunique()}, sesudah: {expanded_df['Matched_SFIA'].nunique()}.\n")


if __name__ == "__main__":
    start_time = time.time()
    program_studies = ["UNAIR_IS", "ITS_IS"]
    MODELS = [
        'SkillNER', 
        'SkillNER_QE',
        'SkillNER_RAKE', 
        'SkillNER_YAKE', 
        'SkillNER_KeyBERT'
    ]

    for ps in program_studies:
        for model in MODELS:
            print(f"\n=== Mapping {ps} dengan {model} ===")
            try:
                map_courses_to_sfia(
                    f"Output3/{ps}/Ekstraksi/skills_extracted_courses_{ps}.xlsx",
                    f"Output3/{ps}/Ekstraksi/skills_extracted_sfia_{ps}.xlsx",
                    model,
                    ps,
                    threshold=0.2
                )
            except Exception as e:
                print(f"[ERROR] {ps} - {model}: {e}")

    print(f"\nTotal waktu: {time.time() - start_time:.2f} detik")
