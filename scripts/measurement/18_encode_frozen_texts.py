"""Portable streaming encoder. CPU/local by default; no paid model calls.

Output/cache is restricted to this script's stage/package root. Input and model
are frozen; chunk checkpoints avoid recomputing successfully stored vectors.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,importlib.metadata
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
from common import ROOT, input_path, setting, out

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,default=out('encoding/full_frozen'));p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--model-dir',type=Path);p.add_argument('--allow-download',action='store_true')
    p.add_argument('--batch-size',type=int);p.add_argument('--block-rows',type=int);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();job=a.job.resolve()
    assert job.is_relative_to(ROOT.resolve()),'Output job must remain inside stage/package root'
    cfg=json.loads((job/'encoder_freeze.json').read_text(encoding='utf-8'))
    inputfile=job/'unique_texts.jsonl';inputhash=digest(inputfile)
    assert inputhash==cfg['inputs_sha256'],'Frozen text input has changed'
    n=int(cfg['unique_texts'])
    for env,rel in [('HF_HOME','cache/huggingface'),('TORCH_HOME','cache/torch'),('TMP','tmp'),('TEMP','tmp')]:
        d=job/rel;d.mkdir(parents=True,exist_ok=True);os.environ[env]=str(d)
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    if not a.allow_download:os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    reuse={};cache_vectors=[]
    for cache in cfg.get('reuse_cache',[]):
        def cachepath(key):
            path=Path(cache[key]);return path if path.is_absolute() else job/path
        previous=json.loads(cachepath('encoder_config').read_text(encoding='utf-8'))
        assert previous['checkpoint_revision']==cfg['checkpoint_revision']
        assert previous['configuration_hashes']==cfg['configuration_hashes']
        array=np.load(cachepath('vectors'),mmap_mode='r');assert array.shape[1]==768
        number=len(cache_vectors);cache_vectors.append(array)
        with cachepath('lookup').open(encoding='utf-8') as lookup:
            for line in lookup:
                row=json.loads(line);index=row['embedding_row'];assert 0<=index<len(array)
                reuse.setdefault(row['text_sha256'],(number,index))
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    if a.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    if a.device=='cuda':assert torch.cuda.is_available(),'CUDA unavailable; no silent CPU fallback for a bulk job'
    local=a.model_dir.resolve() if a.model_dir is not None else Path(cfg.get('local_model') or '__no_local_model__')
    if not local.is_absolute():local=(job/local).resolve()
    if local.is_dir():
        for name,h in cfg['configuration_hashes'].items():assert digest(local/name)==h,'Checkpoint configuration mismatch: '+name
        model=SentenceTransformer(str(local),device=a.device,local_files_only=True,trust_remote_code=False)
    else:
        assert a.allow_download,'Provide the pinned local model or explicitly allow a free public model download'
        model=SentenceTransformer(cfg['model_id'],revision=cfg['checkpoint_revision'],device=a.device,
          cache_folder=str(job/'cache/huggingface'),trust_remote_code=False)
    assert model.max_seq_length==512 and model.get_sentence_embedding_dimension()==768
    pool=model[1]
    if hasattr(pool,'pooling_mode'):assert pool.pooling_mode=='cls'
    else:assert pool.pooling_mode_cls_token is True
    batch=a.batch_size or (128 if a.device=='cuda' else 16)
    block=a.block_rows or (1024 if a.device=='cuda' else 64)
    assert batch>0 and block>0
    statuspath=job/'status.json';start=offset=0
    if statuspath.exists():
        st=json.loads(statuspath.read_text(encoding='utf-8'))
        assert st['input_sha256']==inputhash and st['checkpoint_revision']==cfg['checkpoint_revision']
        start=st['completed_unique_texts'];offset=st['input_byte_offset']
    vectorpath=job/'vectors.npy'
    vectors=np.lib.format.open_memmap(vectorpath,mode='r+' if vectorpath.exists() else 'w+',dtype=np.float32,shape=(n,768))
    parts=job/'token_length_parts';parts.mkdir(exist_ok=True)
    started=time.monotonic();done=start;reused_count=computed_count=0
    versions={name:importlib.metadata.version(name) for name in ['numpy','torch','sentence-transformers','transformers','huggingface-hub']}
    save(job/'runtime_versions.json',{'versions':versions,'device':a.device,'batch_size':batch,'threads':a.threads,
      'checkpoint_revision':cfg['checkpoint_revision'],'paid_api_calls':0})
    print(json.dumps({'status':'ENCODING','input_unique_texts':n,'completed_before':start,'device':a.device},ensure_ascii=False),flush=True)
    if n==0:
        save(statuspath,{'status':'COMPLETE','input_sha256':inputhash,'checkpoint_revision':cfg['checkpoint_revision'],
          'completed_unique_texts':0,'total_unique_texts':0,'input_byte_offset':0,'paid_api_calls':0});return
    with inputfile.open('rb') as f,torch.inference_mode():
        f.seek(offset)
        while done<n:
            records=[]
            for i in range(min(block,n-done)):
                line=f.readline();assert line,'Unexpected end of frozen input'
                r=json.loads(line);assert r['embedding_row']==done+i
                assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
                records.append(r)
            texts=[r['text'] for r in records]
            lengths=model.tokenizer(texts,add_special_tokens=True,truncation=False,padding=False,return_length=True)['length']
            encoded=np.empty((len(records),768),dtype=np.float32);missing=[]
            for i,r in enumerate(records):
                cached=reuse.get(r['text_sha256'])
                if cached is None:missing.append(i)
                else:encoded[i]=cache_vectors[cached[0]][cached[1]]
            while missing:
                try:
                    encoded[missing]=model.encode([texts[i] for i in missing],batch_size=batch,
                      normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch<=1:raise
                    batch=max(1,batch//2);torch.cuda.empty_cache()
                    print(json.dumps({'CUDA_OOM_retry_batch_size':batch}),flush=True)
            reused_count+=len(records)-len(missing);computed_count+=len(missing)
            assert np.isfinite(encoded).all() and np.max(np.abs(np.linalg.norm(encoded,axis=1)-1))<1e-5
            end=done+len(records);vectors[done:end]=encoded;vectors.flush()
            part=parts/(f'{done:09d}_{end:09d}.json')
            save(part,[{'embedding_row':done+i,'tokens':int(v),'truncated':v>512} for i,v in enumerate(lengths)])
            done=end;elapsed=time.monotonic()-started
            state={'status':'COMPLETE' if done==n else 'IN_PROGRESS','input_sha256':inputhash,
              'checkpoint_revision':cfg['checkpoint_revision'],'completed_unique_texts':done,'total_unique_texts':n,
              'input_byte_offset':f.tell(),'seconds_this_run':round(elapsed,2),
              'texts_per_second_this_run':round((done-start)/elapsed,3),'paid_api_calls':0,'device':a.device,
              'reused_vectors_this_run':reused_count,'computed_vectors_this_run':computed_count,'effective_batch_size':batch}
            save(statuspath,state);print(json.dumps(state),flush=True)
    assert done==n

if __name__=='__main__':main()
