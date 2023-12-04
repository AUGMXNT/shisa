import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Dataset provided
data = """
Model	JP AVG	jamp_exact_match	janli_exact_match	jcommonsenseqa_exact_match	jemhopqa_char_f1	jnli_exact_match	jsem_exact_match	jsick_exact_match	jsquad_char_f1	jsts_pearson	jsts_spearman	niilc_char_f1
augmxnt/shisa-base-7b-v1	0.7118	0.3900	0.9600	0.8500	0.4759	0.8200	0.7100	0.7700	0.8557	0.8576	0.8537	0.2868
shisa-base-7b.mega-v1.2	0.7660	0.5800	0.9900	0.8800	0.5241	0.8900	0.7800	0.7600	0.8818	0.8930	0.8817	0.3654
shisa-base-7b.ultra-v0.4	0.7320	0.4800	0.9900	0.8400	0.4682	0.8300	0.7200	0.7500	0.8810	0.8847	0.8750	0.3330
shisa-base-7b.jpslm-instruct-gamma	0.6881	0.3500	0.9300	0.8400	0.4594	0.7700	0.7000	0.6900	0.8582	0.8450	0.8150	0.3114
stabilityai/japanese-stablelm-base-gamma-7b	0.5215	0.4700	0.6000	0.7300	0.4733	0.1700	0.7500	0.2500	0.8603	0.4925	0.4333	0.5068
jpslm-gamma-7b.ultra-v0.4	0.7402	0.4900	0.9400	0.9400	0.5037	0.7200	0.7500	0.7000	0.8839	0.8934	0.8216	0.4999
stabilityai/japanese-stablelm-instruct-gamma-7b	0.5227	0.3900	0.5300	0.8200	0.4691	0.1800	0.5900	0.2300	0.8528	0.6302	0.6010	0.4562
"""

# Loading data into DataFrame
df = pd.read_csv(StringIO(data), sep='\t')

# Creating offset DataFrame for shisa models
shisa_models = df[df['Model'].isin(['augmxnt/shisa-base-7b-v1', 'shisa-base-7b.ultra-v0.4', 'shisa-base-7b.jpslm-instruct-gamma'])].reset_index(drop=True)
shisa_offset_df = shisa_models.drop(columns=['Model', 'JP AVG']).subtract(shisa_models.iloc[0, 2:])
shisa_offset_df['Model'] = shisa_models['Model']
shisa_offset_df = shisa_offset_df.drop(0).reset_index(drop=True)

# Creating offset DataFrame for jpslm models
jpslm_models = df[df['Model'].isin(['jpslm-gamma-7b.ultra-v0.4', 'stabilityai/japanese-stablelm-instruct-gamma-7b', 'stabilityai/japanese-stablelm-base-gamma-7b'])].reset_index(drop=True)
jpslm_offset_df = jpslm_models.drop(columns=['Model', 'JP AVG']).subtract(jpslm_models.iloc[0, 2:])
jpslm_offset_df['Model'] = jpslm_models['Model']
jpslm_offset_df = jpslm_offset_df.drop(0).reset_index(drop=True)

# Plot settings
n_groups = len(shisa_offset_df.columns) - 1
bar_width = 0.35
index = np.arange(n_groups)
colors = {'shisa-base-7b.ultra-v0.4': 'red', 'shisa-base-7b.jpslm-instruct-gamma': 'purple', 'jpslm-gamma-7b.ultra-v0.4': 'red', 'stabilityai/japanese-stablelm-instruct-gamma-7b': 'purple'}

# Creating the first bar plot (shisa models)
fig, ax = plt.subplots(figsize=(12, 6))
for i, row in shisa_offset_df.iterrows():
    ax.bar(index + i * bar_width, row[:-1], bar_width, label=row['Model'], color=colors[row['Model']], alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(index + bar_width / 2, shisa_offset_df.columns[:-1], rotation=45, ha='right')
plt.title('shisa-base-7b-v1 base model w/ ultra-v0.4 vs stablelm-instruct-gamma tune')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
plt.tight_layout()
# plt.show()
dpi = 80
plt.savefig('llm-jp-eval.shisa-base-offset.png', dpi=dpi)

# Creating the second bar plot (jpslm models)
fig, ax = plt.subplots(figsize=(12, 6))
for i, row in jpslm_offset_df.iterrows():
    ax.bar(index + i * bar_width, row[:-1], bar_width, label=row['Model'], color=colors[row['Model']], alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(index + bar_width / 2, jpslm_offset_df.columns[:-1], rotation=45, ha='right')
plt.title('jp-stablelm-gamma base model w/ ultra-v0.4 vs stable-instruct-gamma tune')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
plt.tight_layout()
# plt.show()

dpi = 80
plt.savefig('llm-jp-eval.jpslm-offset.png', dpi=dpi)
