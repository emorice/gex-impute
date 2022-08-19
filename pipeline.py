"""
Gene imputation development pipeline on GTEx data
"""

import urllib.request
import gzip
from contextlib import ExitStack

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc

import galp

step = galp.StepSet()

# Constants
URLS = {
    ## Url of public RNA-seq counts
    'counts': 'https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz',
    ## Url of corresponding meta data, mostly tissue type
    'sample_info': 'https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
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
