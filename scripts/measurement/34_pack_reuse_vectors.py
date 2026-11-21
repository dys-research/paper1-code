"""Copy eligible semantic vectors into the working directory without editing source arrays."""
from common import *
from project_config import input_path
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-rows', type=int, default=65536)
    args = parser.parse_args()
    if args.block_rows <= 0:
        parser.error('--block-rows must be positive')
    job = out('semantic_reuse/all_future_text')
    cfg = json.loads((job / 'reuse_definition.json').read_text(encoding='utf-8'))
    source = input_path('semantic_vectors')
    if str(source) != cfg['source_vector_path']:
        raise ValueError('Semantic vector configuration changed; rebuild metadata first')
    before = source.stat()
    if before.st_size != cfg['source_vector_bytes'] or before.st_mtime_ns != cfg['source_vector_mtime_ns']:
        raise ValueError('Semantic source array changed; rebuild metadata before packing')
    array = np.load(source, mmap_mode='r')
    if list(array.shape) != cfg['source_shape'] or str(array.dtype) != cfg['source_dtype']:
        raise ValueError('Semantic source shape or dtype changed')
    fingerprint = sha(job / 'reuse_definition.json')
    statepath = job / 'vectors_pack_status.json'
    done = 0
    if statepath.exists():
        state = json.loads(statepath.read_text(encoding='utf-8'))
        if state['input_fingerprint'] != fingerprint:
            raise ValueError('Packing inputs changed; use a clean work directory')
        done = state['completed_candidates']
    indices = pq.read_table(job / 'candidate_metadata.parquet', columns=['embedding_row'])['embedding_row'].to_numpy()
    destination = job / 'candidate_vectors.npy'
    if statepath.exists() and state.get('status') == 'COMPLETE':
        focal_destination = job / 'focal_vectors.npy'
        if not destination.exists() or not focal_destination.exists():
            raise ValueError('Complete packing receipt has missing output arrays')
        packed_check = np.load(destination, mmap_mode='r')
        focal_check = np.load(focal_destination, mmap_mode='r')
        focal_n = pq.ParquetFile(job / 'focal_metadata.parquet').metadata.num_rows
        if packed_check.shape != (len(indices), array.shape[1]) or focal_check.shape != (focal_n, array.shape[1]):
            raise ValueError('Complete packing receipt has incompatible output arrays')
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return
    if done and not destination.exists():
        raise ValueError('Packing checkpoint exists but the candidate vector array is missing')
    if destination.exists():
        packed = np.load(destination, mmap_mode='r+')
        if packed.shape != (len(indices), array.shape[1]) or packed.dtype != array.dtype:
            raise ValueError('Existing packed array shape or dtype does not match')
    else:
        packed = np.lib.format.open_memmap(destination, mode='w+', dtype=array.dtype,
                                          shape=(len(indices), array.shape[1]))
    for start in range(done, len(indices), args.block_rows):
        end = min(start + args.block_rows, len(indices))
        packed[start:end] = array[indices[start:end]]
        packed.flush()
        dump('semantic_reuse/all_future_text/vectors_pack_status.json', {
            'status': 'COPYING', 'completed_candidates': end, 'total_candidates': len(indices),
            'input_fingerprint': fingerprint, 'source_readonly': True})
        log(f'Semantic vectors packed: {end}/{len(indices)}')
    query_indices = pq.read_table(job / 'focal_metadata.parquet', columns=['embedding_row'])['embedding_row'].to_numpy()
    # Invalid vectors are preserved. The computation marks invalid focal labels missing
    # and excludes invalid candidate vectors rather than silently altering the corpus.
    np.save(job / 'focal_vectors.npy', np.asarray(array[query_indices]))
    after = source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError('Semantic source changed while packing')
    result = {'status': 'COMPLETE', 'completed_candidates': len(indices), 'total_candidates': len(indices),
              'focal_vectors': len(query_indices), 'input_fingerprint': fingerprint,
              'candidate_file_bytes': destination.stat().st_size,
              'focal_file_bytes': (job / 'focal_vectors.npy').stat().st_size,
              'source_readonly': True, 'source_size_mtime_unchanged': True,
              'large_source_content_hash_performed': False}
    dump('semantic_reuse/all_future_text/vectors_pack_status.json', result)
    finish('34_reuse_vectors_copied', result)


if __name__ == '__main__':
    main()
