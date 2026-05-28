import pandas as pd
import os
from utils.text_preprocessing import preprocess_text
from utils.sfia_processing import transform_sfia_to_long_format

def preprocess_courses_and_sfia(course_file: str, sfia_file: str):
    # Load data kurikulum
    courses_df = pd.read_excel(course_file)
    sfia_df = transform_sfia_to_long_format(sfia_file)

    # Translate & clean LO (Learning Outcomes)
    courses_df['LO_cleaned'] = courses_df['Learning Outcomes'].apply(preprocess_text)

    # Translate & clean course content
    courses_df['course_content_cleaned'] = courses_df['Course Content'].apply(preprocess_text)

    # Translate & clean course description
    if 'Course Description' in courses_df.columns:
        courses_df['course_description_cleaned'] = courses_df['Course Description'].apply(preprocess_text)

    # Process SFIA
    print("Mengubah dan membersihkan deskripsi SFIA...")
    sfia_df['Level_Description_cleaned'] = sfia_df['Level_Description'].apply(preprocess_text)

    return courses_df, sfia_df

if __name__ == '__main__':
    cluster_names = ["UNAIR_IS", "ITS_IS"]

    base_dir = os.path.dirname(os.path.abspath(__file__))  
    data_dir = os.path.join(base_dir, "Data")
    output_base = os.path.join(base_dir, "Output02")

    for cluster_name in cluster_names:
        print(f"\n=== Processing cluster: {cluster_name} ===")

        course_file = os.path.join(data_dir, f"{cluster_name}.xlsx")
        sfia_file   = os.path.join(data_dir, "sfia9_en2025.xlsx")

        # pastikan folder output ada
        output_dir = os.path.join(output_base, cluster_name, "Preprocessing")
        os.makedirs(output_dir, exist_ok=True)

        # jalankan preprocessing
        courses_df, sfia_df = preprocess_courses_and_sfia(course_file, sfia_file)

        # simpan hasil
        courses_df.to_excel(os.path.join(output_dir, f"processed_courses_{cluster_name}.xlsx"), index=False)
        sfia_df.to_excel(os.path.join(output_dir, f"processed_sfia_{cluster_name}.xlsx"), index=False)

        print(f"✔ Data kurikulum {cluster_name} disimpan ke processed_courses_{cluster_name}.xlsx")
        print(f"✔ Data SFIA {cluster_name} disimpan ke processed_sfia_{cluster_name}.xlsx")

