import pandas as pd
import matplotlib.pyplot as plt
import os

cluster_name = "ITS_IS"  # Ganti UNAIR atau ITS

# Semua metrik yang ingin divisualisasikan
METRICS = [
    'Precision (%)',
    'Recall (%)',
    'F1_score (%)',
    'Accuracy (%)'
]

file_tanpa_filtering = (f"Tanpa Filtering/{cluster_name}/Evaluasi/evaluation_results_{cluster_name}.xlsx")

file_dengan_filtering = (f"Dengan Filtering/{cluster_name}/Evaluasi/evaluation_results_{cluster_name}.xlsx")

if not os.path.exists(file_tanpa_filtering):
    raise FileNotFoundError(f"File tidak ditemukan:\n{file_tanpa_filtering}")

if not os.path.exists(file_dengan_filtering):
    raise FileNotFoundError(f"File tidak ditemukan:\n{file_dengan_filtering}")

# READ FILE
df_tanpa = pd.read_excel(file_tanpa_filtering)
df_dengan = pd.read_excel(file_dengan_filtering)

# FILTER EXPANDED
df_tanpa_true = df_tanpa[df_tanpa['Expanded'] == True]
df_tanpa_false = df_tanpa[df_tanpa['Expanded'] == False]

df_dengan_true = df_dengan[df_dengan['Expanded'] == True]
df_dengan_false = df_dengan[df_dengan['Expanded'] == False]

# OUTPUT FOLDER
output_folder = f"{cluster_name}_Evaluation_Comparison/"
os.makedirs(output_folder, exist_ok=True)


# STYLE
plt.style.use('seaborn-v0_8-whitegrid')


# LOOP SEMUA METRIC
for METRIC in METRICS:

    print(f"\nMembuat visualisasi: {METRIC}")

    # CREATE FIGURE
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    # LEFT PLOT - TANPA FILTERING
    ax1 = axes[0]

    # Expanded=True
    ax1.plot(
        df_tanpa_true['Model'],
        df_tanpa_true[METRIC],
        marker='o',
        linestyle='-',
        linewidth=2,
        label='Expanded=True'
    )

    for i, txt in enumerate(df_tanpa_true[METRIC]):
        ax1.annotate(
            f'{txt:.3f}',
            (
                df_tanpa_true['Model'].iloc[i],
                df_tanpa_true[METRIC].iloc[i]
            ),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    # Expanded=False
    ax1.plot(
        df_tanpa_false['Model'],
        df_tanpa_false[METRIC],
        marker='x',
        linestyle='--',
        linewidth=2,
        label='Expanded=False'
    )

    for i, txt in enumerate(df_tanpa_false[METRIC]):
        ax1.annotate(
            f'{txt:.3f}',
            (
                df_tanpa_false['Model'].iloc[i],
                df_tanpa_false[METRIC].iloc[i]
            ),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center'
        )

    ax1.set_title(
        f'Tanpa Filtering\n{METRIC} Comparison',
        fontsize=14,
        fontweight='bold'
    )

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel(METRIC, fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    
    # RIGHT PLOT - DENGAN FILTERING
    ax2 = axes[1]

    # Expanded=True
    ax2.plot(
        df_dengan_true['Model'],
        df_dengan_true[METRIC],
        marker='o',
        linestyle='-',
        linewidth=2,
        label='Expanded=True'
    )

    for i, txt in enumerate(df_dengan_true[METRIC]):
        ax2.annotate(
            f'{txt:.3f}',
            (
                df_dengan_true['Model'].iloc[i],
                df_dengan_true[METRIC].iloc[i]
            ),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    # Expanded=False
    ax2.plot(
        df_dengan_false['Model'],
        df_dengan_false[METRIC],
        marker='x',
        linestyle='--',
        linewidth=2,
        label='Expanded=False'
    )

    for i, txt in enumerate(df_dengan_false[METRIC]):
        ax2.annotate(
            f'{txt:.3f}',
            (
                df_dengan_false['Model'].iloc[i],
                df_dengan_false[METRIC].iloc[i]
            ),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center'
        )

    ax2.set_title(
        f'Dengan Filtering\n{METRIC} Comparison',
        fontsize=14,
        fontweight='bold'
    )

    ax2.set_xlabel('Model', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    
    # MAIN TITLE
    fig.suptitle(
        (
            f'Comparison of {METRIC} '
            f'Before and After Expansion ({cluster_name})'
        ),
        fontsize=18,
        fontweight='bold'
    )

    # LAYOUT
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # SAVE FIGURE
    output_file = (
        f"{output_folder}"
        f"{METRIC.lower()}_comparison_{cluster_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    print(f"Gambar disimpan: {output_file}")

    # SHOW
    plt.show()

    # CLOSE FIGURE
    plt.close()