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
problems = ['Portfolio']

show_results = True
show_dro_comparison = True

# If show_results == True
CUSTOM_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2"]

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

sns.set_theme(style="darkgrid", context="talk", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def prepare_data(df):

    df_long = df.melt(id_vars=['norm', 'eta'],
                      value_vars=['P_2NN', 'P_SRF'],
                      var_name='Method_Raw',
                      value_name='Performance')

    def clean_name(name):
        if '2NN' in str(name): return '2NN'
        if 'SRF' in str(name): return 'SRF'
        return name

    df_long['Method'] = df_long['Method_Raw'].apply(clean_name)

    df_long['Plot_Group'] = df_long.apply(lambda row: f"{row['Method']}_{row['eta']}", axis=1)

    target_eta = sorted(df_long['eta'].unique())

    return df_long, target_eta

def Boxplot_2NN_and_SRF(df_pf, folder_path, file_name):

    data, target_eta = prepare_data(df_pf)

    data['Performance'] = data['Performance'] / 100.0

    if len(target_eta) > len(CUSTOM_COLORS):
        colors_to_use = CUSTOM_COLORS * 2
    else:
        colors_to_use = CUSTOM_COLORS

    eta_colors = {eta: color for eta, color in zip(target_eta, colors_to_use)}

    hue_order = []
    palette_list = []

    for eta in target_eta:
        hue_order.append(f"2NN_{eta}")
        palette_list.append(eta_colors[eta])
        hue_order.append(f"SRF_{eta}")
        palette_list.append(eta_colors[eta])

    fig, ax = plt.subplots(figsize=(15, 7))

    sns.boxplot(
        data=data,
        x='norm',
        y='Performance',
        hue='Plot_Group',
        hue_order=hue_order,
        palette=palette_list,
        width=0.8,
        linewidth=1.2,
        fliersize=2,
        ax=ax
    )

    box_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]

    box_patches.sort(key=lambda p: p.get_path().vertices[:, 0].mean())

    group_size = 2 * len(target_eta)

    for i, patch in enumerate(box_patches):
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch.set_hatch(None)

        intra_group_index = i % group_size

        if intra_group_index % 2 == 0:
            patch.set_linestyle('--')
        else:
            patch.set_linestyle('-')

    mean_srf = data[data['Method'] == 'SRF']['Performance'].mean()
    mean_2nn = data[data['Method'] == '2NN']['Performance'].mean()

    line_srf_color = '#D55E00'
    line_2nn_color = '#0072B2'

    ax.axhline(y=mean_srf, color=line_srf_color, linestyle='--', linewidth=1.5, alpha=0.9)
    ax.axhline(y=mean_2nn, color=line_2nn_color, linestyle='-.', linewidth=1.5, alpha=0.9)

    ax.set_xlabel(r'$p$-Causal-SDRO', fontsize=16)
    ax.set_ylabel('Out-of-sample Performance', fontsize=16)

    norms = sorted(data['norm'].unique())
    ax.set_xticklabels([f"$p={n}$" for n in norms])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(labelsize=13)

    y_min, y_max = data['Performance'].min(), data['Performance'].max()
    ax.set_ylim(-1, 1)

    legend_handles = []

    legend_handles.append(mpatches.Patch(visible=False, label=r'Factor ($\omega$)'))

    for eta in target_eta:
        label_str = r"$\omega={}$".format(eta)
        patch = mpatches.Patch(facecolor=eta_colors[eta], edgecolor='black', linewidth=1, label=label_str)
        legend_handles.append(patch)

    legend_handles.append(mpatches.Patch(visible=False, label=""))

    legend_handles.append(mpatches.Patch(visible=False, label='Method'))

    h_2nn = mpatches.Patch(facecolor='white', edgecolor='black', linestyle='--', linewidth=1.5, label='2NN (Dashed)')
    legend_handles.append(h_2nn)

    h_srf = mpatches.Patch(facecolor='white', edgecolor='black', linestyle='-', linewidth=1.5, label='SRF (Solid)')
    legend_handles.append(h_srf)

    legend_handles.append(mpatches.Patch(visible=False, label=""))

    legend_handles.append(mpatches.Patch(visible=False, label='Average Performance'))

    h_mean_2nn = mlines.Line2D([], [], color=line_2nn_color, linestyle='-.', label=f'2NN Mean ({mean_2nn:.1%})')
    legend_handles.append(h_mean_2nn)

    h_mean_srf = mlines.Line2D([], [], color=line_srf_color, linestyle='--', label=f'SRF Mean ({mean_srf:.1%})')
    legend_handles.append(h_mean_srf)

    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=12,
        labelspacing=0.6,
        borderaxespad=0
    )

    plt.tight_layout()

    if not os.path.exists(folder_path): os.makedirs(folder_path)
    save_path = os.path.join(folder_path, f"{file_name}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Portfolio results saved: {save_path}")
    plt.close(fig)

def Boxplot_DRO(df, folder_path, file_name_no_ext, compared_models):

    cols = ['Parameters', 'Instance'] + ALL_DRO_MODELS
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].copy()

    data_melt = pd.melt(
        data,
        id_vars=['Parameters', 'Instance'],
        value_vars=[m for m in ALL_DRO_MODELS if m in valid_cols],
        var_name='Model',
        value_name='Performance'
    )

    data_melt = data_melt[data_melt['Model'].isin(compared_models)]
    data_melt['Performance'] = data_melt['Performance'] / 100.0

    fig, ax = plt.subplots(figsize=(14, 7))

    palette_list = [MODEL_COLOR_MAP[m] for m in compared_models]

    sns.boxplot(
        data=data_melt,
        x='Parameters',
        y='Performance',
        hue='Model',
        hue_order=compared_models,
        palette=palette_list,
        width=0.8,
        linewidth=1.2,
        fliersize=2,
        ax=ax
    )

    box_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
    for patch in box_patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)
        patch.set_linestyle('-')

    ax.set_xlabel('Factor ($\omega$)', fontsize=16)
    ax.set_ylabel('Out-of-sample Performance', fontsize=16)

    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(labelsize=13)

    legend_handles = []
    for m in compared_models:
        label_name = m.replace('P_', '')
        patch = mpatches.Patch(
            facecolor=MODEL_COLOR_MAP[m],
            edgecolor='black',
            linewidth=1,
            label=label_name
        )
        legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=11,
        title="DRO Models"
    )

    plt.tight_layout()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save_path = os.path.join(folder_path, f"{file_name_no_ext}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"DRO Comparison plot saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    for problem in problems:

        if show_results == True:

            # Please ensure that this file exists
            read_file_path = f'../Outputs/Outputs_CSDRO/DRO_{problem}_all_results.xlsx'

            output_folder_path = f'../Outputs/Outputs_Visualization/{problem}_visualizations'
            output_file = f"{problem}_Results_Boxplot."

            df_pf = pd.read_excel(read_file_path, engine='openpyxl')

            Boxplot_2NN_and_SRF(df_pf, output_folder_path, output_file)

        if show_dro_comparison == True:
            read_file_path = f'../Outputs/Outputs_DRO_Comparison/DRO_Compare_{problem}_results.xlsx'
            output_folder_path = f'../Outputs/Outputs_Visualization/{problem}_visualizations'
            output_file = f"{problem}_DRO_Comparison_Boxplot"

            if os.path.exists(read_file_path):
                df_dro = pd.read_excel(read_file_path, engine='openpyxl')
                Boxplot_DRO(df_dro, output_folder_path, output_file, Compared_DRO_models)
            else:
                print(f"Comparison file missing: {read_file_path}")