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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_filename = 'DRO_Newsvendor_all_results.xlsx'
output_formats = ('pdf', 'svg')
plot_dx_values = [3, 5, 10]
smooth_nm_values = [5, 10]

# Basic settings
problems = ['Newsvendor']

# Visualize the performance of different decision rules.
show_results = True
# Visualize the performance of different DRO models.
show_dro_comparison = True
# Visualize Smooth against Causal-SDRO separately.
show_smooth_comparison = True

# If show_results == True
CUSTOM_COLORS = ["#E69F00", "#56B4E9", "#CC79A7", "#009E73"]

# If show_dro_comparison == True
# Select compared models in the boxplots
Compared_DRO_models = ['P_1-Causal-SDRO', 'P_2-Causal-SDRO', 'P_1-SDRO', 'P_2-SDRO',
    'P_1-Causal-WDRO', 'P_2-Causal-WDRO', 'P_KL-DRO']

Compared_Smooth_models = ['P_1-Causal-SDRO', 'P_2-Causal-SDRO', 'P_1-Smooth', 'P_2-Smooth']
SMOOTH_HATCH_MAP = {model: '///' if 'Causal-SDRO' in model else ''
                    for model in Compared_Smooth_models}

CUSTOM_COLORS_DRO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#00B050", "#E7298A"
]
ALL_DRO_MODELS = [
    'P_1-Causal-SDRO', 'P_2-Causal-SDRO', 'P_1-SDRO', 'P_2-SDRO',
    'P_1-Causal-WDRO', 'P_2-Causal-WDRO', 'P_KL-DRO', 'P_1-Smooth', 'P_2-Smooth'
]

# Construct a mapping dict
MODEL_COLOR_MAP = dict(zip(ALL_DRO_MODELS, CUSTOM_COLORS_DRO))

p_norms = [1, 2]
sns.set_theme(style="darkgrid", context="talk", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def save_figure(fig, folder_path, file_name_no_ext):
    os.makedirs(folder_path, exist_ok=True)
    for extension in output_formats:
        save_path = os.path.join(folder_path, f'{file_name_no_ext}.{extension}')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.close(fig)


def prepare_data(df):

    mask = df['Model'].apply(lambda x: 'SRF' in str(x) or 'TNN' in str(x) or '2NN' in str(x))
    df_comp = df[mask].copy()

    def clean_name(name):
        if 'SRF' in str(name): return 'SRF'
        if 'TNN' in str(name) or '2NN' in str(name): return '2NN'
        return name

    df_comp['Method'] = df_comp['Model'].apply(clean_name)

    target_N = [20, 40, 80, 100]
    target_Dx = plot_dx_values
    df_comp = df_comp[df_comp['N'].isin(target_N) & df_comp['D_X'].isin(target_Dx)]

    if df_comp.empty: return None

    df_comp['Plot_Group'] = df_comp.apply(lambda row: f"{row['Method']}_{row['D_X']}", axis=1)

    return df_comp

def Boxplot_2NN_and_SRF(df, folder_path, file_name_no_ext):
    data = prepare_data(df)
    if data is None: return

    data['Prescriptiveness'] = data['Prescriptiveness'] / 100.0

    target_Dx = plot_dx_values

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

        intra_group_index = i % len(hue_order)

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

    save_figure(fig, folder_path, file_name_no_ext)


def Boxplot_DRO(df, folder_path, file_name_no_ext, compared_models):

    df = df[df['D_X'].isin(plot_dx_values)].copy()
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
            fontsize=10
        )

        plt.tight_layout()

        save_figure(fig, folder_path, f'{file_name_no_ext}_{file_suffix}')

    for dx in sorted(df['D_X'].dropna().unique()):
        df_fixed_dx = data_melt[data_melt['D_X'] == dx].sort_values(by='N')
        draw_single_boxplot(df_fixed_dx, 'N', 'Sample Size ($N$)', f'Fixed_Dx_{dx:g}')


def Boxplot_Smooth(df, folder_path, file_name_no_ext):
    """One row per selected n_m; hatching distinguishes Causal-SDRO from Smooth."""
    required = ['N', 'D_X', 'n_m', 'Instance'] + Compared_Smooth_models
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f'Missing columns for Smooth comparison: {missing}')
    data = df.loc[df['D_X'].isin(plot_dx_values) & df['n_m'].isin(smooth_nm_values),
                  required].copy()
    # Keep all four methods paired on the same available experimental instances.
    data = data.dropna(subset=Compared_Smooth_models)
    nm_values = sorted(data['n_m'].dropna().unique())
    n_values = sorted(data['N'].dropna().unique())
    dx_values = sorted(data['D_X'].dropna().unique())
    if data.empty:
        print('Skip Smooth comparison: no available cases.')
        return

    fig, axes = plt.subplots(len(nm_values), 1,
                             figsize=(18, 3.5 * len(nm_values) + 1), squeeze=False)
    dimension_width = 0.84 / len(dx_values)
    method_spacing = 0.20 * dimension_width
    box_width = 0.17 * dimension_width
    for row, n_m in enumerate(nm_values):
        ax = axes[row, 0]
        for n_index, n in enumerate(n_values):
            for dx_index, dx in enumerate(dx_values):
                center = n_index + (dx_index - (len(dx_values) - 1) / 2) * dimension_width
                group = data[(data['n_m'] == n_m) & (data['N'] == n) & (data['D_X'] == dx)]
                for method_index, method in enumerate(Compared_Smooth_models):
                    values = group[method] / 100.0
                    if values.empty:
                        continue
                    artists = ax.boxplot(
                        values, positions=[center + (method_index - 1.5) * method_spacing],
                        widths=box_width, patch_artist=True, manage_ticks=False,
                        boxprops={'facecolor': MODEL_COLOR_MAP[method], 'edgecolor': 'black',
                                  'linewidth': 1, 'hatch': SMOOTH_HATCH_MAP[method]},
                        medianprops={'color': 'black', 'linewidth': 1},
                        whiskerprops={'color': 'black', 'linewidth': 0.9},
                        capprops={'color': 'black', 'linewidth': 0.9},
                        flierprops={'marker': 'o', 'markersize': 1.8, 'markeredgecolor': 'black'})
                    artists['boxes'][0].set_gid(f'nm{n_m:g}_N{n:g}_dx{dx:g}_{method}')
                ax.text(center, -0.055, f'$d_x={dx:g}$', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=11)
        ax.set_xticks(range(len(n_values)))
        ax.set_xticklabels([f'{n:g}' for n in n_values], fontsize=14)
        ax.tick_params(axis='x', pad=30)
        ax.set_xlim(-0.55, len(n_values) - 0.45)
        ax.set_ylabel('Out-of-sample Performance\n' + f'($n_m={n_m:g}$)', fontsize=14)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(False, axis='x')
        for boundary in range(len(n_values) - 1):
            ax.axvline(boundary + 0.5, color='white', linewidth=1.5)
    axes[-1, 0].set_xlabel('Sample Size ($N$)', fontsize=16, labelpad=10)
    handles = [mpatches.Patch(facecolor=MODEL_COLOR_MAP[method], edgecolor='black',
                              hatch=SMOOTH_HATCH_MAP[method],
                              label=method.replace('P_', '')) for method in Compared_Smooth_models]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 0.995),
               ncol=4, frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.10, hspace=0.40)
    save_figure(fig, folder_path, file_name_no_ext)


if __name__ == "__main__":
    for problem in problems:
        if show_results == True:

            # Please ensure that this file exists
            read_file_path = os.path.join(PROJECT_DIR, 'Outputs/Outputs_CSDRO', results_filename)

            df = pd.read_excel(read_file_path, engine='openpyxl')

            output_folder_path = os.path.join(PROJECT_DIR,
                                              f'Outputs/Outputs_Visualization/{problem}_visualizations')
            # Keep different n_m experiments separate; support older result files too.
            nm_groups = df.groupby('n_m', sort=True) if 'n_m' in df.columns else [(None, df)]
            for n_m, df_nm in nm_groups:
                for p in p_norms:
                    df_norm = df_nm[df_nm['Norm'] == p].copy()
                    if n_m is None:
                        output_file = f"{problem}_Results_Boxplot_p={p}"
                    else:
                        output_file = f"{problem}_Results_Boxplot_p{p}_nm{n_m:g}"

                    Boxplot_2NN_and_SRF(df_norm, output_folder_path, output_file)

        if show_dro_comparison == True or show_smooth_comparison == True:

            read_file_path = os.path.join(PROJECT_DIR,
                                          f'Outputs/Outputs_DRO_Comparison/DRO_Compare_{problem}_results.xlsx')

            output_folder_path = os.path.join(PROJECT_DIR,
                                              f'Outputs/Outputs_Visualization/{problem}_visualizations')
            output_file = f"{problem}_DRO_Comparison_Boxplot"

            df = pd.read_excel(read_file_path, engine='openpyxl')

            # Other DRO comparisons retain one set of figures per n_m.
            if show_dro_comparison == True:
                if 'n_m' in df.columns:
                    for n_m in sorted(df['n_m'].unique()):
                        df_nm = df[df['n_m'] == n_m].copy()
                        Boxplot_DRO(df_nm, output_folder_path, f"{output_file}_nm{n_m:g}",
                                    Compared_DRO_models)
                else:
                    Boxplot_DRO(df, output_folder_path, output_file, Compared_DRO_models)
            # One Smooth figure covers all N and dx for the selected n_m values.
            if show_smooth_comparison == True:
                Boxplot_Smooth(df, output_folder_path,
                               f"{problem}_Smooth_Comparison_Boxplot_All_N_Dx_nm")
