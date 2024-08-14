"""
Gene imputation development pipeline on GTEx data
"""

import urllib.request
import gzip
from contextlib import ExitStack

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc

import numpy as np

import plotly.graph_objects as go
import plotly_template

from galp import step, view
import gemz
import gemz_galp.models
import gemz.plots

# Constants
URLS = {
    ## Url of public RNA-seq counts
    'counts': (
        'https://storage.googleapis.com/gtex_analysis_v8/'
        'rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz'
        ),
    ## Url of corresponding meta data, mostly tissue type
    'sample_info': (
        'https://storage.googleapis.com/gtex_analysis_v8/'
        'annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
        )
}

## Buffer size for streaming raw RNA-seq
BUF_SIZE = 10 * 2**20

@step(vtag='+lz4')
def gex_counts_table(_galp):
    """
    Download the gene expression table in arrow format under a new file
    """
    path = _galp.new_path()

    with ExitStack() as stack:
        # Start HTTP request
        response = stack.enter_context(
            urllib.request.urlopen(URLS['counts'])
            )
        # Wrap response in file object with gz decompression
        uncompressed = stack.enter_context(
                gzip.open(response)
                )

        # Build Arrow csv reader
        reader = pacsv.open_csv(
            uncompressed,
            read_options=pacsv.ReadOptions(skip_rows=2, block_size=BUF_SIZE),
            parse_options=pacsv.ParseOptions(delimiter='\t'),
        )

        # Get batch #1 and infer schema
        batch = reader.read_next_batch()

        # Initialize writer
        writer = stack.enter_context(
            pa.RecordBatchFileWriter(path, batch.schema,
                options=pa.ipc.IpcWriteOptions(compression='lz4')
                )
            )

        writer.write(batch)

        i = 1
        while True:
            if not i % 10:
                print(f'Gex table: imported {i * BUF_SIZE / 2 ** 20} MB')
            try:
                batch = reader.read_next_batch()
                writer.write(batch)
            except StopIteration:
                break
            i += 1

    return path

@step
def gex_sample_info_table():
    """
    Download the sample information to an arrow table.

    It's small so we directly check it in the store.
    """
    with urllib.request.urlopen(URLS['sample_info']) as response:
        return pacsv.read_csv(response,
                parse_options=pacsv.ParseOptions(delimiter='\t')
                )

downloads = [
    gex_counts_table,
    gex_sample_info_table,
    ]

@step(vtag=1)
def gex_tissue_counts_table(tissue_name, gex_counts_table, gex_sample_info_table):
    """
    Filter a single tissue from the counts table, and checks the result in store.
    """
    sampids = (gex_sample_info_table
            .filter(pc.field('SMTSD') == tissue_name)
            ['SAMPID']
            .to_pylist()
            )

    out_batches = []
    with pa.ipc.open_file(gex_counts_table) as reader:
        names = []
        for i in range(reader.num_record_batches):
            in_batch = reader.get_batch(i)
            if not i:
                names = [ name
                    for name in ['Name', 'Description'] + sampids
                    if name in in_batch.schema.names
                    ]
            out_batches.append(
                pa.RecordBatch.from_arrays([
                    in_batch[name]
                    for name in names
                    ], names=names)
                )
        return pa.Table.from_batches(out_batches)

@step(vtag='swapdims')
def gex_tissue_counts(gex_tissue_counts_table):
    """
    Split the table into two meta data tables and a numpy array

    The array has shape (sample_count, gene_count), so the transpose of the
    table shape
    """
    gene_info = gex_tissue_counts_table.select(('Name', 'Description'))

    table = gex_tissue_counts_table.drop(('Name', 'Description'))

    sample_info = pa.Table.from_pydict({'sample': pa.array(table.column_names)})

    # See https://github.com/apache/arrow/pull/36242 , we used (sample,gene)
    # because it used to be the default ; now we have a transpose for compat
    return sample_info, gene_info, np.array(table).T

@step
def gex_insample_transformed_counts(gex_tissue_counts, transformation):
    """
    Apply in-sample, across-genes transformation
    """
    sample_info, gene_info, data = gex_tissue_counts

    if transformation == 'raw':
        return sample_info, gene_info, data

    if transformation == 'cpm':
        # Across genes = dim -1
        library_size = data.sum(axis=-1, keepdims=True)
        sample_info = sample_info.append_column('library_size',
                pa.array(np.squeeze(library_size, axis=-1)))
        return (sample_info, gene_info,
            data / library_size * 1e6
            )

    # To be added: TMM, QN
    raise NotImplementedError(transformation)

@step(vtag='+dict')
def gex_tissue_shape(gex_tissue_counts):
    """
    Number of genes and samples for a given tissue
    """
    sample_info, gene_info, data = gex_tissue_counts
    assert len(sample_info) == data.shape[0]
    assert len(gene_info) == data.shape[1]

    return {
        'gene_count': len(gene_info),
        'sample_count': len(sample_info)
        }

_FC = 5
#step.bind(gex_fold_count=_FC)

@step
def gex_tissue_cv_sample_masks(gex_tissue_shape, gex_fold_count):
    """
    Generate cross-validation masks for a given tissue
    """
    return gemz.utils.cv_masks(gex_fold_count, gex_tissue_shape['sample_count'])

@step(items=_FC)
def gex_tissue_qn(gex_insample_transformed_counts, gex_tissue_cv_sample_masks):
    """
    Quantile normalize across-samples, in-gene, on top of possible across-genes
    pre-transformations, for each cv-fold
    """
    sample_info, gene_info, data = gex_insample_transformed_counts

    qn_data_list = gemz.utils.quantile_normalize(
        data,
        gex_tissue_cv_sample_masks[:, :, None], # Bcast mask along genes
        axis=0, # Across samples = dim 0
        )

    return [
        (sample_info, gene_info, qn_data)
        for qn_data in qn_data_list
        ]

@step(vtag='-bool')
def gex_tissue_expressed_gene_masks(gex_tissue_counts,
        gex_tissue_cv_sample_masks,
        count_threshold=6, prop_threshold=0.2):
    """
    Compute masks of genes to keep based on a "at least 20% with at least 6 reads" fixed
    threshold
    """
    _sample_info, _gene_info, data_sg = gex_tissue_counts
    sample_masks_ms = gex_tissue_cv_sample_masks

    above_thr_sg = 1 * (data_sg >= count_threshold)
    num_above_thr_mg = sample_masks_ms @ above_thr_sg
    sample_count_m1 = sample_masks_ms.sum(axis=-1, keepdims=True)
    above_prop_thr_mg = num_above_thr_mg >= prop_threshold * sample_count_m1

    return above_prop_thr_mg

@step
def gex_tissue_fold(
        gex_tissue_qn_indexed,
        fold_index,
        gex_tissue_expressed_gene_masks,
        gex_tissue_cv_sample_masks
        ):
    """
    Train/test split along samples of transformed gene expression, with gene
    expression filters
    """
    sample_info, gene_info, data = gex_tissue_qn_indexed

    gene_mask = gex_tissue_expressed_gene_masks[fold_index]
    sample_mask = gex_tissue_cv_sample_masks[fold_index]

    gene_info = gene_info.filter(gene_mask)
    data = data[:, gene_mask]

    train = data[sample_mask, :]
    test = data[~sample_mask, :]

    sample_info = sample_info.append_column('is_train', pa.array(sample_mask))

    return {
        'train': train,
        'test': test,
        'gene_info': gene_info,
        'sample_info': sample_info,
        }

# Test configuration
#step.bind(tissue_name='Whole Blood', transformation='cpm')
#step.bind(gex_tissue_qn_indexed=gex_tissue_qn[0], fold_index=0)

models=[
    (
        spec,
        gemz_galp.models.fit_eval(
            spec,
            gex_tissue_fold,
            'iRSS'
            )
    )
    for spec in ([
        {'model': 'linear'},
        {'model': 'peer', 'n_factors': 60},
        {'model': 'peer', 'n_factors': 60, 'reestimate_precision': True},
        {'model': 'igmm', 'n_groups': 2},
        {'model': 'lscv_precision_target'},
        {'model': 'lscv_free_diagonal'},
        {'model': 'lscv_free_diagonal', 'scale': None},
        ] + [
        {'model': 'cv', 'inner': {'model': inner}, 'loss_name': 'GEOM', 'grid_max': gmax}
        for inner, gmax in [
            ('linear_shrinkage', None),
            ('svd', None),
            ('cmk', None),
            ('igmm', 30), # 30 mixture components
        #    ('peer', 100), # 100 peer factors
            ]
        ])
    ]

#step.bind(models=models)

#step.bind(model_losses=[
#    (spec, fit_eval['loss'])
#    for spec, fit_eval in models
#    ])

@step(vtag='0.2 renaming')
def model_gene_r2s(
        gex_tissue_fold,
        model_losses
        ):
    """
    Per gene residual share of variance for all evaluated models
    """
    test_variances = np.var(gex_tissue_fold['test'], axis=0)
    num_test_samples = gex_tissue_fold['test'].shape[0]

    return pa.concat_tables(
        gex_tissue_fold['gene_info']
            .append_column('model', pa.array(np.full(
                len(gex_tissue_fold['gene_info']),
                gemz.models.get_name(spec)
                )))
            .append_column('r2', pa.array(
                loss / (test_variances * num_test_samples)
                ))
        for spec, loss in model_losses
        )

@view
def hist_linear_r2(model_gene_r2s):
    """
    Histogram of per-gene difficulty as measured by perf of reference model
    """
    return go.Figure([
                go.Histogram(x=
                    model_gene_r2s
                        .filter(pc.field('model') == 'linear')
                        ['r2']
                    )
            ], {
                'xaxis.title': 'Linear model residual R^2',
                'yaxis.title': 'Number of genes',
                'width': 800,
                }
            )

#step.bind(ref_model='linear')
#step.bind(alt_model='igmm/2')

@view
def vs_r2(model_gene_r2s, ref_model, alt_model):
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
                    marker={'size': 2}
                    ),
                go.Scattergl(
                    x=trend[ref_model], y=trend[alt_model] / trend[ref_model], mode='lines',
                    hovertext=[
                    f'{100 * l:.6g} - {100 *u:.6g} %'
                    for l, u in
                    zip(trend.index, [*trend.index[1:], 1.0])
                    ],
                    name='median'),
                go.Scattergl(
                    x=r2_df[ref_model],
                    y=np.ones_like(r2_df[ref_model]),
                    mode='lines',
                    name='baseline'
                    )
            ], {
                'title': f'{alt_model} vs. {ref_model}',
                'xaxis.title': f'Reference model ({ref_model}) residual R²',
                'xaxis.type': 'log',
                'yaxis.title': f'Relative alternative model ({alt_model}) residual R²',
                'width': 1000,
                'height': 800,
                }
            )

@view
def all_r2(model_gene_r2s, ref_model):
    """
    Scatter of per-gene difficulty as measured by perf wrt reference model, for all models in compact form.
    """
    highlights = {'cv/igmm'}

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
                        {'color': 'blue', 'width': 1}
                        if model in highlights
                        else {'color': 'black', 'width': 1}
                        ),
                    opacity = 1. if model in highlights else 0.2,
                    hovertext=[
                    f'{100 * l:.6g} - {100 *u:.6g} %'
                    for l, u in
                    zip(trends.index, [*trends.index[1:], 1.0])
                    ],
                    name=model, showlegend=(model in highlights)
                    )
                for model in trends if model not in (ref_model, 'abs_ref')
            ] + [
                go.Scatter(
                    x=trends['abs_ref'], y=trends[ref_model],
                    mode='lines',
                    line={'color': 'red', 'width': 1},
                    name='baseline'
                    )
            ], {
                'title': f'Medians of all models vs. {ref_model}',
                'xaxis.title': f'Reference model ({ref_model}) residual R²',
                'xaxis.type': 'log',
                'yaxis.title': f'Relative alternative model residual R²',
                'legend.title': 'Model',
                'width': 600, #1000,
                'height': 400, #800,
                'margin': {'t': 40},
                }
            )

#step.bind(cv_model=next(
#            (spec, cfe)
#            for spec, cfe in models
#            if spec['model'] == 'cv'
#            if spec['inner']['model'] == 'igmm'
#            )
#        )

@view
def cv_plot(cv_model):
    """
    Cross-validation curve for bound model
    """
    spec, cfe = cv_model

    fig, = gemz.plots.plot_cv(spec, cfe['fit'])
    fig = fig.update_layout(title=f'CV results for {gemz.models.get_name(spec)}')
    return fig
