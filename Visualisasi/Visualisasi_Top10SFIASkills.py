import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib.gridspec as gridspec


# CONFIGURATION
CLUSTER_NAME = 'ITS_IS'  # Ganti UNAIR atau ITS
TOP_SKILLS = 10

# List metode yang ingin dibandingkan
METHODS = [
    'SkillNER',
    'SkillNER_QE',
    'SkillNER_RAKE',
    'SkillNER_YAKE',
    'SkillNER_KeyBERT',
]

# Folder sumber
BASE_FOLDERS = {
    'Tanpa Filtering': 'Tanpa Filtering',
    'Dengan Filtering': 'Dengan Filtering'
}

# Folder output
OUTPUT_FOLDER = f'{CLUSTER_NAME}_Radar_Comparison/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# RADAR PLOT FUNCTION
def plot_single_radar(ax, df, title, top_n=10, color='blue'):

    if 'Matched_SFIA' not in df.columns:
        ax.set_title(f"{title}\n(Kolom tidak ditemukan)", color='red')
        return []

    skill_counts = (
        df['Matched_SFIA']
        .value_counts()
        .nlargest(top_n)
    )

    if skill_counts.empty:
        ax.set_title(f"{title}\n(Tidak ada data)", color='red')
        return []

    labels_raw = skill_counts.index.tolist()
    values = skill_counts.values.tolist()

    num_vars = len(labels_raw)

    angles = np.linspace(
        0,
        2 * np.pi,
        num_vars,
        endpoint=False
    ).tolist()

    # Menutup radar chart
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    # Plot radar
    ax.plot(
        angles_closed,
        values_closed,
        linewidth=2.5,
        linestyle='solid',
        color=color,
        zorder=3
    )

    ax.fill(
        angles_closed,
        values_closed,
        color=color,
        alpha=0.15,
        zorder=2
    )

    # Label A, B, C
    abjad_labels = list(string.ascii_uppercase)[:num_vars]

    ax.set_xticks(angles)
    ax.set_xticklabels(
        abjad_labels,
        size=18,
        fontweight='bold'
    )

    # Skala Y
    max_val = max(values) if values else 1

    tick_step = (
        np.ceil(max_val / 4 / 5) * 5
        if max_val > 10
        else np.ceil(max_val / 4)
    )

    if tick_step == 0:
        tick_step = 1

    yticks = np.arange(
        tick_step,
        max_val + tick_step,
        tick_step
    )

    ax.set_yticks(yticks)

    ax.set_yticklabels(
        [str(int(y)) for y in yticks],
        color='grey',
        size=10
    )

    ax.set_ylim(0, max_val * 1.15)

    # Style
    ax.spines['polar'].set_visible(False)

    ax.grid(
        axis='y',
        linestyle='--',
        color='black',
        alpha=0.8,
        zorder=1
    )

    ax.set_title(
        title,
        size=20,
        y=1.1,
        weight='bold'
    )

    # Legend text
    legend_info = [
        f"{abjad_labels[i]}. {labels_raw[i]} ({values[i]})"
        for i in range(num_vars)
    ]

    return legend_info


# MAIN VISUALIZATION FUNCTION
def visualize_comparison(cluster_name, methods, top_n=10):
    print(f"Membuat visualisasi radar untuk cluster: {cluster_name}")
    for method in methods:
        print(f"\nProcessing method: {method}")
        
        # FILE PATH
        file_tanpa = (
            f"Tanpa Filtering/{cluster_name}/Mapping/expanded_mapping_cosine_{method}_{cluster_name}.xlsx"
        )

        file_dengan = (
            f"Dengan Filtering/{cluster_name}/Mapping/expanded_mapping_cosine_{method}_{cluster_name}.xlsx"
        )

        # CHECK FILE EXISTENCE
        if not os.path.exists(file_tanpa):
            print(f"File tidak ditemukan: {file_tanpa}")
            continue

        if not os.path.exists(file_dengan):
            print(f"File tidak ditemukan: {file_dengan}")
            continue

        # READ DATA
        try:
            df_tanpa = pd.read_excel(file_tanpa)
            df_dengan = pd.read_excel(file_dengan)

        except Exception as e:
            print(f"Gagal membaca file untuk {method}: {e}")
            continue

        # CREATE FIGURE
        fig = plt.figure(
            figsize=(24, 15),
            constrained_layout=True
        )

        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[10, 5],
            hspace=0.3,
            wspace=0.1
        )

        radar_ax1 = fig.add_subplot(gs[0, 0], polar=True)
        radar_ax2 = fig.add_subplot(gs[0, 1], polar=True)

        legend_ax1 = fig.add_subplot(gs[1, 0])
        legend_ax2 = fig.add_subplot(gs[1, 1])

        legend_ax1.axis('off')
        legend_ax2.axis('off')

        
        # PLOT TANPA FILTERING
        legend_text1 = plot_single_radar(
            radar_ax1,
            df_tanpa,
            f'{method}\n(Tanpa Filtering)',
            top_n=top_n,
            color='#1f77b4'
        )

        legend_ax1.text(
            0.05,
            0.95,
            '\n'.join(legend_text1),
            fontsize=18,
            va='top',
            ha='left',
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=0.8',
                fc='aliceblue',
                ec='lightgray',
                lw=1
            )
        )

        # PLOT DENGAN FILTERING
        legend_text2 = plot_single_radar(
            radar_ax2,
            df_dengan,
            f'{method}\n(Dengan Filtering)',
            top_n=top_n,
            color='#ff7f0e'
        )

        legend_ax2.text(
            0.05,
            0.95,
            '\n'.join(legend_text2),
            fontsize=18,
            va='top',
            ha='left',
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=0.8',
                fc='oldlace',
                ec='lightgray',
                lw=1
            )
        )

        
        # TITLE
        fig.suptitle(
            f'Top {top_n} Mapped SFIA Skills Comparison\n{method} - {cluster_name}',
            fontsize=26,
            fontweight='bold'
        )

        # SAVE FIGURE
        output_file = (
            f'{OUTPUT_FOLDER}'
            f'radar_comparison_{method}_{cluster_name}.png'
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.3,
            facecolor='white'
        )

        plt.close(fig)

        print(f'Visualisasi disimpan: {output_file}')



# MAIN
if __name__ == '__main__':

    visualize_comparison(
        cluster_name=CLUSTER_NAME,
        methods=METHODS,
        top_n=TOP_SKILLS
    )
