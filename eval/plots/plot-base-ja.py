import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.lines import Line2D
import numpy as np
import random

# Load the Excel file
file_path = 'shisa-plot.xlsx'  # Replace with your file path
df = pd.read_excel(file_path, sheet_name='Base Models')

# Selecting the relevant columns for the spider plot
categories = df.columns[12:23]  # Columns I:W
labels = df['Model']
values = df[categories]

# Replace NaNs with column means
values = values.apply(lambda x: x.fillna(x.mean()), axis=0)

# Add the first value to the end of each row to complete the loop
values_loop = np.hstack((values, values.iloc[:,0].values.reshape(-1,1)))

# Correcting the angles to match the number of categories
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Ensure the loop is closed by repeating the first value

# Adjusting the plot with the new requirements
fig, ax = plt.subplots(figsize=(18, 10), subplot_kw=dict(polar=True))

# Adjust the subplot position to be left-aligned
fig.subplots_adjust(left=0, right=0.6)

category_colors = [
    {"category": "MC", "color": (0, 0, 0.5, 1)},
    {"category": "NLI", "color": (0, 0, 0.5, 1)},
    {"category": "QA", "color": (0, 0, 0.5, 1)},
    {"category": "RC", "color": (0, 0, 0.5, 1)},
    {"category": "jamp_exact_match", "color": (0.4, 0.15, 0, 1)},
    {"category": "janli_exact_match", "color": (0.4, 0.15, 0, 1)},
    {"category": "jcommonsenseqa_exact_match", "color": (0.4, 0.15, 0, 1)},
    {"category": "jemhopqa_char_f1", "color": (0.4, 0.15, 0, 1)},
    {"category": "jnli_exact_match", "color": (0.4, 0.15, 0, 1)},
    {"category": "jsem_exact_match", "color": (0.4, 0.15, 0, 1)},
    {"category": "jsick_exact_match", "color": (0.4, 0.15, 0, 1)},
    {"category": "jsquad_char_f1", "color": (0.4, 0.15, 0, 1)},
    {"category": "jsts_pearson", "color": (0.4, 0.15, 0, 1)},
    {"category": "jsts_spearman", "color": (0.4, 0.15, 0, 1)},
    {"category": "niilc_char_f1", "color": (0.4, 0.15, 0, 1)},
]


# Draw one axe per variable + add labels with individual colors
for i, angle in enumerate(angles[:-1]):
    category_color = next(item['color'] for item in category_colors if item['category'] == categories[i])
    ax.text(angle, 1.1, categories[i], horizontalalignment='center', size=10, fontweight='bold', color=category_color)

# Outline
spine_color = (0.8, 0.8, 0.8, 0.8)
for spine in ax.spines.values():
    spine.set_edgecolor(spine_color)

# Remove radial lines
ax.yaxis.grid(color=(0.8, 0.8, 0.8, 0.3), linestyle='-')
ax.xaxis.grid(color=(0.8, 0.8, 0.8, 0.3), linestyle='-')
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.50, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
plt.ylim(0, 1)

# Model names and their manually assigned RGBA colors
# https://rgbcolorpicker.com/0-1
linewidth = 4
models = [
    # Mistral
    {
        "name": "stabilityai/japanese-stablelm-base-beta-70b", 
        "label": "Japanese-StableLM-Base-Beta-70B", 
        "color": (0.5, 0.7, 0.9, 0.2),
        "visible": False,
        "alpha": 0.1,
    },
    {
        "name": "stabilityai/japanese-stablelm-base-gamma-7b", 
        "label": "Japanese Stable LM Base Gamma 7B",
        "color": (0.5, 0.5, 1, 1.0),
        "visible": False,
        "alpha": 0.0,
    },
    {
        "name": "mistralai/mistral-7b-v0.1", 
        "label": "Mistral 7B",
        "color": (0.33, 0.88, 0.75, 1.0),
        "visible": False,
        "alpha": 0.0,
    },

    # JA Tokenizer
    {
        "name": "llm-jp/llm-jp-13b-v1.0", 
        "color": (0.8, 0.6, 0.3, 0.2),
        "visible": True,
        "alpha": 0.1,
    },
    {
        "name": "cyberagent/calm2-7b",
        "color": (0.6, 0.5, 0.9, 1.0),
        "visible": True,
        "alpha": 0.0,
    },
    {
        "name": "elyza/ELYZA-japanese-Llama-2-7b-fast", 
        "color": (0.5, 0.7, 1.0, 1.0),
        "visible": True,
        "alpha": 0.0,
    },  
    {
        "name": "rinna/youri-7b", 
        "color": (0.5, 0.9, 0.6, 1.0),
        "visible": True,
        "alpha": 0.0,
    },

    # Our Model
    {
        "name": "augmxnt/shisa-base-7b-v1", 
        "label": "shisa-base-7b-v1",
        "color": (1.0, 0.3, 0.4, 1.0),
        "visible": True,
        "alpha": 0.0,
    },

]

# Function to find values from values_loop based on the model name
find_values = lambda name: values_loop[labels == name].flatten()
    
# Create custom legend handles considering the "visible" entry
legend_handles = []
legend_labels = []

# Plotting each model using the custom array
for model in models:
    if model['visible']:
        model['values'] = find_values(model['name'])

        # Get rid of negative values which screws up chart rendering - looking at you llm-jp-13b 
        model['values'] = np.maximum(model['values'], 0.00)

        print(model['values'])
        ax.plot(angles, model['values'], linewidth=linewidth, linestyle='solid', label=model['name'], color=model['color'])
        ax.fill(angles, model['values'], alpha=model['alpha'], color=model['color'])

        # Make dots a bit transparent to see overlay
        if model['color'][3] < 0.75:
            alpha = model['color'][3]
        else:
            alpha = 0.75

        ax.scatter(angles, model['values'], color=model['color'], alpha=alpha, s=80)  # s is the size of the dot

        # Add to legend handles and labels
        legend_handles.append(Line2D([0], [0], color=model['color'], lw=8))  # Adjust lw as needed
        legend_labels.append(model['name'])

# Add a title
plt.title('llm-jp-eval score', size=20, color='black', y=1.1)

# Apply custom handles to the legend
plt.legend(handles=legend_handles, labels=legend_labels, loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=18, handleheight=2.5, frameon=False)


dpi = 80
plt.savefig('llm-jp-eval.ja.png', dpi=dpi)

# Show the plot
# plt.show()

