import pandas as pd
import matplotlib.pyplot as plt

df_unair = pd.read_excel("Dengan Filtering/UNAIR_IS/Ekstraksi/skills_extracted_courses_UNAIR_IS.xlsx")
df_its = pd.read_excel("Dengan Filtering/ITS_IS/Ekstraksi/skills_extracted_courses_ITS_IS.xlsx")

skill_count_columns = [
    'SkillNER_after',
    'SkillNER_QE_after',
    'SkillNER_RAKE_after',
    'SkillNER_YAKE_after',
    'SkillNER_KeyBERT_after'
]

results = {
    'Model': [],
    'Cluster': [],
    'Total': [],
    'Average': []
}

def calculate_cluster_stats(df, cluster_name):
    for col in skill_count_columns:
        total = df[col].sum()
        average = df[col].mean()
        model = col.replace('_count', '')
        results['Model'].append(model)
        results['Cluster'].append(cluster_name)
        results['Total'].append(total)
        results['Average'].append(average)

calculate_cluster_stats(df_unair, 'UNAIR IS')
calculate_cluster_stats(df_its, 'ITS IS')

results_df = pd.DataFrame(results)

# ------------------ Plot Total ------------------
fig_total, ax_total = plt.subplots(figsize=(10, 6))

bar_width = 0.35
x = range(len(skill_count_columns))

unair_totals = results_df[results_df['Cluster'] == 'UNAIR IS']['Total']
its_totals = results_df[results_df['Cluster'] == 'ITS IS']['Total']
labels = results_df[results_df['Cluster'] == 'UNAIR IS']['Model']

x_pos = list(x)
x_pos2 = [p + bar_width for p in x_pos]

bars_unair = ax_total.bar(x_pos, unair_totals, width=bar_width, label='UNAIR IS', color='skyblue')
bars_its = ax_total.bar(x_pos2, its_totals, width=bar_width, label='ITS IS', color='salmon')

ax_total.set_xlabel('Skill Extraction Models')
ax_total.set_ylabel('Total Value')
ax_total.set_title('Comparison of Total Value of Skill Extraction Methods (UNAIR IS vs ITS IS)')
ax_total.set_xticks([p + bar_width / 2 for p in x_pos])
ax_total.set_xticklabels(labels, rotation=45, ha='right')
ax_total.grid(axis='y', linestyle='--', alpha=0.7)
ax_total.legend()

for bar in bars_unair + bars_its:
    yval = bar.get_height()
    ax_total.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# ------------------ Plot Average ------------------
fig_avg, ax_avg = plt.subplots(figsize=(10, 6))

unair_avg = results_df[results_df['Cluster'] == 'UNAIR IS']['Average']
its_avg = results_df[results_df['Cluster'] == 'ITS IS']['Average']

bars_cs_avg = ax_avg.bar(x_pos, unair_avg, width=bar_width, label='UNAIR IS', color='skyblue')
bars_is_avg = ax_avg.bar(x_pos2, its_avg, width=bar_width, label='ITS IS', color='salmon')

ax_avg.set_xlabel('Skill Extraction Models')
ax_avg.set_ylabel('Average Score')
ax_avg.set_title('Comparison of Average Value of Skill Extraction Methods (UNAIR IS vs ITS IS)')
ax_avg.set_xticks([p + bar_width / 2 for p in x_pos])
ax_avg.set_xticklabels(labels, rotation=45, ha='right')
ax_avg.grid(axis='y', linestyle='--', alpha=0.7)
ax_avg.legend()

for bar in bars_cs_avg + bars_is_avg:
    yval = bar.get_height()
    ax_avg.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()