"""
    This file visualizes all numerical results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os

# Basic settings
problems = ['Inventory']

show_results = True
show_dro_comparison = True

# If show_results == True
CUSTOM_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]

# If show_dro_comparison == True
# Select compared models in the boxplots
Compared_DRO_models = ['P_1-Causal-SDRO', 'P_2-Causal-SDRO', 'P_1-SDRO', 'P_2-SDRO',
    'P_1-Causal-WDRO', 'P_2-Causal-WDRO', 'P_KL-DRO']

CUSTOM_COLORS_DRO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7"
]
ALL_DRO_MODELS = [
    'P_1-Causal-SDRO', 'P_2-Causal-SDRO', 'P_1-SDRO', 'P_2-SDRO',
    'P_1-Causal-WDRO', 'P_2-Causal-WDRO', 'P_KL-DRO'
]
# Construct a mapping dict
MODEL_COLOR_MAP = dict(zip(ALL_DRO_MODELS, CUSTOM_COLORS_DRO))

fixed_n = 400
fixed_dx = 20

p_norms = [1, 2]
sns.set_theme(style="darkgrid", context="talk", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def prepare_data(df):

    mask = df['Model'].apply(lambda x: 'SRF' in str(x) or 'TNN' in str(x) or '2NN' in str(x))
    df_comp = df[mask].copy()

    def clean_name(name):
        if 'SRF' in str(name): return 'SRF'
        if 'TNN' in str(name) or '2NN' in str(name): return '2NN'
        return name

    df_comp['Method'] = df_comp['Model'].apply(clean_name)

    target_N = [100, 200, 400, 800]
    target_Dx = [3, 5, 10, 20]
    df_comp = df_comp[df_comp['N'].isin(target_N) & df_comp['D_X'].isin(target_Dx)]

    if df_comp.empty: return None

    df_comp['Plot_Group'] = df_comp.apply(lambda row: f"{row['Method']}_{row['D_X']}", axis=1)

    return df_comp

def Boxplot_2NN_and_SRF(df, folder_path, file_name_no_ext):
    data = prepare_data(df)
    if data is None: return

    data['Prescriptiveness'] = data['Prescriptiveness'] / 100.0

    target_Dx = [3, 5, 10, 20]

    dx_colors = {dx: color for dx, color in zip(target_Dx, CUSTOM_COLORS)}

    hue_order = []
    palette_list = []

    for dx in target_Dx:
        # 2NN
        hue_order.append(f"2NN_{dx}")
        palette_list.append(dx_colors[dx])
        # SRF
        hue_order.append(f"SRF_{dx}")
        palette_list.append(dx_colors[dx])

    fig, ax = plt.subplots(figsize=(15, 7))

    sns.boxplot(
        data=data,
        x='N',
        y='Prescriptiveness',
        hue='Plot_Group',
        hue_order=hue_order,
        palette=palette_list,
        width=0.75,
        linewidth=1.2,
        fliersize=2,
        ax=ax
    )

    box_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
    box_patches.sort(key=lambda p: p.get_path().vertices[:, 0].mean())

    for i, patch in enumerate(box_patches):
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch.set_hatch(None)

        intra_group_index = i % 8

        if intra_group_index % 2 == 0:
            # 2NN
            patch.set_linestyle('--')
        else:
            # SRF
            patch.set_linestyle('-')

    mean_srf = data[data['Method'] == 'SRF']['Prescriptiveness'].mean()
    mean_2nn = data[data['Method'] == '2NN']['Prescriptiveness'].mean()

    line_srf_color = '#D55E00'
    line_2nn_color = '#0072B2'

    ax.axhline(y=mean_srf, color=line_srf_color, linestyle='--', linewidth=1.5, alpha=0.9)
    ax.axhline(y=mean_2nn, color=line_2nn_color, linestyle='-.', linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Sample Size (N)', fontsize=16)
    ax.set_ylabel('Out-of-sample Performance', fontsize=16)
    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(labelsize=13)

    legend_handles = []

    legend_handles.append(mpatches.Patch(visible=False, label='Covariate Dimension'))

    for dx in target_Dx:
        label_str = r"$d_x={}$".format(dx)
        patch = mpatches.Patch(facecolor=dx_colors[dx], edgecolor='black', linewidth=1, label=label_str)
        legend_handles.append(patch)

    legend_handles.append(mpatches.Patch(visible=False, label=""))  # Spacer

    legend_handles.append(mpatches.Patch(visible=False, label='Method'))

    h_2nn = mpatches.Patch(facecolor='white', edgecolor='black', linestyle='--', linewidth=1.5, label='2NN (Dashed)')
    legend_handles.append(h_2nn)

    h_srf = mpatches.Patch(facecolor='white', edgecolor='black', linestyle='-', linewidth=1.5, label='SRF (Solid)')
    legend_handles.append(h_srf)

    legend_handles.append(mpatches.Patch(visible=False, label=""))  # Spacer

    legend_handles.append(mpatches.Patch(visible=False, label='Average Performance'))

    h_mean_2nn = mlines.Line2D([], [], color=line_2nn_color, linestyle='-.', label=f'2NN Mean ({mean_2nn:.1%})')
    legend_handles.append(h_mean_2nn)

    h_mean_srf = mlines.Line2D([], [], color=line_srf_color, linestyle='--', label=f'SRF Mean ({mean_srf:.1%})')
    legend_handles.append(h_mean_srf)

    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        labelspacing=0.6,
        borderaxespad=0
    )

    plt.tight_layout()

    fig_dir = folder_path
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    save_path = os.path.join(folder_path, f"{file_name_no_ext}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Save to {save_path}")
    plt.close(fig)


def Boxplot_DRO(df, folder_path, file_name_no_ext, fixed_n, fixed_dx, compared_models):

    cols = ['N', 'D_X', 'Instance'] + ALL_DRO_MODELS
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].copy()

    data_melt = pd.melt(
        data,
        id_vars=['N', 'D_X', 'Instance'],
        value_vars=[m for m in ALL_DRO_MODELS if m in valid_cols],
        var_name='Model',
        value_name='Performance'
    )

    data_melt = data_melt[data_melt['Model'].isin(compared_models)]
    data_melt['Performance'] = data_melt['Performance'] / 100.0

    def draw_single_boxplot(plot_data, x_col, x_label, file_suffix):
        if plot_data.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        palette_list = [MODEL_COLOR_MAP[m] for m in compared_models]

        sns.boxplot(
            data=plot_data,
            x=x_col,
            y='Performance',
            hue='Model',
            hue_order=compared_models,
            palette=palette_list,
            width=0.7,
            linewidth=1.2,
            fliersize=2,
            ax=ax
        )

        box_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        for patch in box_patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_hatch(None)
            patch.set_linestyle('-')

        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel('Out-of-sample Performance', fontsize=16)
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.tick_params(labelsize=13)

        legend_handles = []
        for m in compared_models:
            label_name = m.replace('P_', '')
            patch = mpatches.Patch(facecolor=MODEL_COLOR_MAP[m], edgecolor='black', linewidth=1, label=label_name)
            legend_handles.append(patch)

        ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=10,
            title="Methods"
        )

        plt.tight_layout()

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = os.path.join(folder_path, f"{file_name_no_ext}_{file_suffix}.pdf")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close(fig)

    df_fixed_n = data_melt[data_melt['N'] == fixed_n].copy()
    df_fixed_n.sort_values(by='D_X', inplace=True)
    draw_single_boxplot(df_fixed_n, 'D_X', 'Covariate Dimension ($d_x$)', f"Fixed_N_{fixed_n}")

    df_fixed_dx = data_melt[data_melt['D_X'] == fixed_dx].copy()
    df_fixed_dx.sort_values(by='N', inplace=True)
    draw_single_boxplot(df_fixed_dx, 'N', 'Sample Size ($N$)', f"Fixed_Dx_{fixed_dx}")

if __name__ == "__main__":
    for problem in problems:
        if show_results == True:

            # Please ensure that this file exists
            read_file_path = f'../Outputs/Outputs_CSDRO/DRO_{problem}_all_results.xlsx'

            df = pd.read_excel(read_file_path, engine='openpyxl')

            for p in p_norms:
                df_norm = df[df['Norm'] == p].copy()

                output_folder_path = f'../Outputs/Outputs_Visualization/{problem}_visualizations'
                output_file = f"{problem}_Results_Boxplot_p={p}"

                Boxplot_2NN_and_SRF(df_norm, output_folder_path, output_file)

        if show_dro_comparison == True:

            read_file_path = f'../Outputs/Outputs_DRO_Comparison/DRO_Compare_{problem}_results.xlsx'

            output_folder_path = f'../Outputs/Outputs_Visualization/{problem}_visualizations'
            output_file = f"{problem}_DRO_Comparison_Boxplot"

            df = pd.read_excel(read_file_path, engine='openpyxl')

            # Function name
            Boxplot_DRO(df, output_folder_path, output_file, fixed_n, fixed_dx, Compared_DRO_models)