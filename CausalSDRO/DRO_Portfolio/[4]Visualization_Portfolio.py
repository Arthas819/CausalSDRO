"""
    Visualize the DRO and smoothing comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import os
import math

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_formats = ('pdf', 'svg')
smooth_nm_values = [5, 10]
omega_values = [0.25, 0.5, 0.75, 1.0]

# Basic settings
problems = ['Portfolio']

show_dro_comparison = True
show_smooth_comparison = True

# If show_dro_comparison == True
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


def performance_limits(values):
    values = pd.Series(values).replace([float('inf'), -float('inf')], float('nan')).dropna()
    if values.empty:
        return -1.0, 1.0
    return min(-1.0, math.floor(values.min() * 4) / 4), max(1.0, math.ceil(values.max() * 4) / 4)


def set_performance_axis(ax, limits, tick_step=0.25):
    lower, upper = limits
    ax.set_ylim(lower - 0.05, upper + 0.05)
    ax.set_yticks([i * tick_step for i in range(math.ceil(lower / tick_step),
                                               math.floor(upper / tick_step) + 1)])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def prepare_comparison_data(df):
    data = df.copy()
    # The workbook stores eta in Parameters; figure labels retain omega.
    eta_column = 'eta' if 'eta' in data.columns else 'Parameters'
    data['eta'] = pd.to_numeric(data[eta_column], errors='raise')
    data = data.loc[data['eta'].isin(omega_values)].copy()
    keys = ['n_m', 'eta', 'Instance']
    if data[keys].isna().any().any():
        raise ValueError('Missing Portfolio comparison case keys.')
    if data.duplicated(keys).any():
        raise ValueError('Duplicate Portfolio (n_m, eta, Instance) cases.')
    return data


def Boxplot_DRO(df, folder_path, file_name_no_ext, compared_models, axis_limits=None):

    data = prepare_comparison_data(df)
    if data.empty:
        return
    if data['n_m'].nunique() != 1:
        raise ValueError('Each Portfolio DRO figure must contain one n_m.')
    data = data[['eta', 'Instance'] + compared_models].copy()
    eta_values = sorted(data['eta'].unique())

    data_melt = pd.melt(
        data,
        id_vars=['eta', 'Instance'],
        value_vars=compared_models,
        var_name='Model',
        value_name='Performance'
    )

    data_melt['Performance'] = data_melt['Performance'] / 100.0
    data_melt = data_melt.dropna(subset=['Performance'])
    if data_melt.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    palette_list = [MODEL_COLOR_MAP[m] for m in compared_models]

    sns.boxplot(
        data=data_melt,
        x='eta',
        order=eta_values,
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

    ax.set_xlabel(r'Factor ($\omega$)', fontsize=16)
    ax.set_ylabel('Out-of-sample Performance', fontsize=16)

    set_performance_axis(ax, axis_limits if axis_limits is not None
                          else performance_limits(data_melt['Performance']))
    ax.set_xticks(range(len(eta_values)))
    ax.set_xticklabels([f'{eta:g}' for eta in eta_values])
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
        fontsize=11
    )

    plt.tight_layout()

    save_figure(fig, folder_path, file_name_no_ext)


def Boxplot_Smooth(df, folder_path, file_name_no_ext):
    data = prepare_comparison_data(df.loc[df['n_m'].isin(smooth_nm_values)])
    missing = [column for column in Compared_Smooth_models if column not in data.columns]
    if missing:
        raise ValueError(f'Missing columns for Portfolio Smooth comparison: {missing}')
    # Compare the four methods on the same available trading dates.
    data = data.dropna(subset=Compared_Smooth_models)
    if data.empty:
        print('Skip Smooth comparison: no available cases.')
        return
    nm_values = sorted(data['n_m'].unique())
    eta_values = sorted(data['eta'].unique())
    limits = performance_limits(data[Compared_Smooth_models].to_numpy().ravel() / 100)
    fig, axes = plt.subplots(len(nm_values), 1,
                             figsize=(14, 3.5 * len(nm_values) + 1), squeeze=False)
    for row, n_m in enumerate(nm_values):
        ax = axes[row, 0]
        for eta_index, eta in enumerate(eta_values):
            group = data[(data['n_m'] == n_m) & (data['eta'] == eta)]
            for method_index, method in enumerate(Compared_Smooth_models):
                values = group[method] / 100.0
                if values.empty:
                    continue
                artists = ax.boxplot(
                    values, positions=[eta_index + (method_index - 1.5) * 0.18],
                    widths=0.15, patch_artist=True, manage_ticks=False,
                    boxprops={'facecolor': MODEL_COLOR_MAP[method], 'edgecolor': 'black',
                              'linewidth': 1, 'hatch': SMOOTH_HATCH_MAP[method]},
                    medianprops={'color': 'black', 'linewidth': 1},
                    whiskerprops={'color': 'black', 'linewidth': 0.9},
                    capprops={'color': 'black', 'linewidth': 0.9},
                    flierprops={'marker': 'o', 'markersize': 1.8, 'markeredgecolor': 'black'})
                artists['boxes'][0].set_gid(f'nm{n_m:g}_eta{eta:g}_{method}')
        ax.set_xticks(range(len(eta_values)))
        ax.set_xticklabels([f'{eta:g}' for eta in eta_values], fontsize=14)
        ax.set_xlim(-0.55, len(eta_values) - 0.45)
        ax.set_ylabel('Out-of-sample Performance\n' + f'($n_m={n_m:g}$)', fontsize=14)
        set_performance_axis(ax, limits, tick_step=0.5)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(False, axis='x')
        for boundary in range(len(eta_values) - 1):
            ax.axvline(boundary + 0.5, color='white', linewidth=1.5)
    axes[-1, 0].set_xlabel(r'Factor ($\omega$)', fontsize=16, labelpad=10)
    handles = [mpatches.Patch(facecolor=MODEL_COLOR_MAP[method], edgecolor='black',
                              hatch=SMOOTH_HATCH_MAP[method],
                              label=method.replace('P_', '')) for method in Compared_Smooth_models]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 0.995),
               ncol=4, frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.105, right=0.99,
                        top=0.88 if len(nm_values) == 1 else 0.94,
                        bottom=0.08, hspace=0.25)
    save_figure(fig, folder_path, file_name_no_ext)


def plot_comparison(df):
    data = prepare_comparison_data(df)
    if data.empty:
        print('Skip Portfolio comparison: no cases after filtering omega.')
        return
    folder = os.path.join(PROJECT_DIR, 'Outputs/Outputs_Visualization/Portfolio_visualizations')
    if show_dro_comparison:
        # Use one common vertical scale across n_m, including recorded outliers.
        limits = performance_limits(data[Compared_DRO_models].to_numpy().ravel() / 100)
        for n_m, group in data.groupby('n_m', sort=True):
            Boxplot_DRO(group, folder, f'Portfolio_DRO_Comparison_Boxplot_nm{n_m:g}',
                        Compared_DRO_models, axis_limits=limits)
    if show_smooth_comparison:
        Boxplot_Smooth(data, folder, 'Portfolio_Smooth_Comparison_Boxplot_All_omega_nm')


if __name__ == '__main__':
    if show_dro_comparison or show_smooth_comparison:
        result_path = os.path.join(PROJECT_DIR,
                      'Outputs/Outputs_DRO_Comparison/DRO_Compare_Portfolio_results.xlsx')
        print(f'Portfolio comparison data: {result_path}')
        plot_comparison(pd.read_excel(result_path, sheet_name='All_Results'))
