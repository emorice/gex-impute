"""
Plots of imputation results
"""

import os
import time
import argparse

import pyarrow as pa
import pyarrow.compute as pc

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import galp
import gemz

from gex_impute import template
import pipeline

def hist_r2(model_gene_r2s, ref_model):
    """
    Histogram of per-gene difficulty as measured by perf of reference model
    """
    return go.Figure([
                go.Histogram({
                    'x': model_gene_r2s
                        .filter(pc.field('model') == ref_model)
                        ['r2'],
                    }),
            ], {
                'xaxis.title': 'Residual R²',
                'yaxis.title': 'Number of genes',
                }
            )

def vs_r2(model_gene_r2s: pa.Table, ref_model: str, alt_model: str):
    """
    Scatter of per-gene difficulty as measured by perf wrt reference model
    """
    r2_df = model_gene_r2s.to_pandas()

    r2_df = r2_df.pivot(index=['Name', 'Description'], columns='model', values='r2')

    desc = r2_df.index.to_frame()['Description']

    n_bins = 20
    trend = (
        r2_df[[ref_model, alt_model]]
        .sort_values(ref_model)
        .assign(cdf=
            np.floor(n_bins * np.arange(len(r2_df)) / len(r2_df))
            / n_bins
            )
        .groupby('cdf')
        .median()
        )

    return go.Figure([
                go.Scattergl(
                    x=r2_df[ref_model],
                    y=r2_df[alt_model] / r2_df[ref_model],
                    hovertext=desc,
                    mode='markers',
                    marker={'size': 2},
                    opacity=.8,
                    showlegend=False,
                ),
                go.Scattergl(
                    x=trend[ref_model],
                    y=trend[alt_model] / trend[ref_model],
                    mode='lines',
                    #line={'width': 1.5},
                    hovertext=[
                        f'{100 * l:.6g} - {100 *u:.6g} %'
                        for l, u in zip(trend.index, [*trend.index[1:], 1.0])
                    ],
                    name='median',
                    showlegend=False,
                ),
                go.Scattergl(
                    x=r2_df[ref_model],
                    y=np.ones_like(r2_df[ref_model]),
                    mode='lines',
                    #line={'width': 1.5},
                    name=f'baseline ({ref_model})',
                    showlegend=False,
                )
            ], {
                #'title': f'{alt_model} vs. {ref_model}',
                'xaxis.title': f'Reference model ({ref_model}) residual R²',
                'xaxis.type': 'log',
                'yaxis': {
                    'title': f'Relative alternative model ({alt_model}) residual R²',
                    'rangemode': 'tozero'
                },
                #'margin': {'t': 40},
                }
            )

def all_r2(model_gene_r2s: pa.Table, ref_model: str, highlights: set[str]):
    """
    Scatter of per-gene difficulty as measured by perf wrt reference model, for
    all models in compact form.
    """

    r2_df = (
        model_gene_r2s
        .to_pandas()
        .pivot(index=['Name', 'Description'], columns='model', values='r2')
        .sort_values(ref_model)
        )

    nr2_df = r2_df.apply(lambda s: s / r2_df[ref_model])
    nr2_df['abs_ref'] = r2_df[ref_model]

    n_bins = 20
    trends = (
        nr2_df
        .assign(cdf=
            np.floor(n_bins * np.arange(len(nr2_df)) / len(nr2_df))
            / n_bins
            )
        .groupby('cdf')
        .median()
        )

    return go.Figure([
                go.Scatter(
                    x=trends['abs_ref'], y=trends[model], mode='lines',
                    line=(
                        {'color': 'blue'}#, 'width': 1}
                        if model in highlights
                        else {'color': 'black'}#, 'width': 1}
                        ),
                    opacity = 1. if model in highlights else 0.2,
                    hovertext=[
                    f'{100 * l:.6g} - {100 *u:.6g} %'
                    for l, u in
                    zip(trends.index, [*trends.index[1:], 1.0])
                    ],
                    name=model,
                    showlegend=False, #(model in highlights)
                    )
                for model in trends if model not in (ref_model, 'abs_ref')
            ] + [
                go.Scatter(
                    x=trends['abs_ref'], y=trends[ref_model],
                    mode='lines',
                    line={'color': 'red'},#, 'width': 1},
                    name=f'baseline ({ref_model})',
                    showlegend=False,
                    )
            ], {
                #'title': f'Medians of all models vs. {ref_model}',
                'xaxis.title': f'Reference model ({ref_model}) residual R²',
                'xaxis.type': 'log',
                'xaxis.tickmode': 'array',
                'xaxis.tickvals': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                'xaxis.exponentformat': 'none',
                'yaxis.title': 'Relative alternative model residual R²',
                #'legend.title': 'Model',
                #'margin': {'t': 40},
                }
            )

def median_r2_bars(r2s: pa.Table) -> go.Figure:
    """
    High level summary of results
    """
    model_median_r2 = (
        r2s
        .to_pandas()
        .groupby('model', as_index=False)
        ['r2']
        .median()
        .sort_values('r2', ascending=False)
    )
    return go.Figure([
        go.Bar({
            'orientation': 'h',
            'x': model_median_r2['r2'],
            'y': model_median_r2['model'],
        })
        ], {
            'margin': {'l': 140},
            'yaxis.showline': False,
            'xaxis.title': 'Median gene normalized RMSE',
        }
    )

def cv_plot(spec, grid) -> go.Figure:
    """
    Cross-validation curve for bound model
    """
    specs, losses = zip(*grid)

    axis, = gemz.models.get(spec['inner']['model']).cv.get_grid_axes(specs)

    loss_df = pd.DataFrame(
            {axis['name']: axis['values']}
            ).assign(loss=losses)

    return go.Figure(
            data=[
                go.Scatter(
                    x=loss_df[axis['name']],
                    y=loss_df['loss'],
                    mode='lines+markers',
                    showlegend=False,
                    ),
                ],
            layout={
                'xaxis': {
                    'title': f'Number of {axis["name"].lower()}',
                    'type': 'log' if axis['log'] else 'linear'
                    },
                'yaxis': { 'title': 'Cross-validation loss'},
                }
            )

def write_to(fig: go.Figure, name: str, output: str) -> None:
    """
    Export figure
    """
    fig.write_image(os.path.join(output, f'{name}.svg'))

def export_all_plots(store: str, output: str) -> None:
    """
    Run all the graph generating code and create svg files
    """
    os.makedirs(output, exist_ok=True)

    t_fold = pipeline.get_tissue_fold('Whole Blood', 0)
    specs = pipeline.get_specs()

    fits, t_r2s = pipeline.get_model_gene_r2s(t_fold, specs)
    fits_by_name = {
            gemz.models.get_name(spec): (spec, fit)
            for spec, fit in fits
            }
    start = time.time()
    print('Loading results...', end='', flush=True)
    r2s = galp.run(t_r2s, store=store)
    print(f' OK ({time.time() - start:.0g}s)', flush=True)

    write = lambda fig, name: write_to(fig, name, output)

    write(median_r2_bars(r2s), 'all_median_r2s')
    write(hist_r2(r2s, 'cv/svd'), 'hist_svd_r2')

    svd_spec, svd_cfe = fits_by_name['cv/svd2']
    svd_grid = galp.run(pipeline.extract_cv(
        svd_cfe['fit']), store=store)
    svd_fig = cv_plot(svd_spec, svd_grid)
    write(svd_fig.update_layout({'xaxis.type': 'linear'}), 'cv_svd')
    write(svd_fig.update_layout({
        'xaxis.type': 'linear', 'yaxis.range': [5160, 5190]
        }), 'cv_svd_zoom')

def main():
    """Entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument('store', help='Task result store')
    parser.add_argument('output', help='Task result store')
    args = parser.parse_args()

    export_all_plots(args.store, args.output)
