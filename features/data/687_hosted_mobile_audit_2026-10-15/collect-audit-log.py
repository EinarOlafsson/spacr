import base64,gzip,hashlib,json,pathlib,sys
root=pathlib.Path(__file__).parent
raw=(root/'audit.log').read_bytes()
events=[]
for line in raw.decode().splitlines():
    at=line.find('{"type":')
    if at<0: continue
    try: event=json.loads(line[at:])
    except json.JSONDecodeError: continue
    events.append(event)
assert len([x for x in events if x.get('type')=='context'])==1
reports={}
for meta in (x for x in events if x.get('type')=='report'):
    chunks=sorted((x for x in events if x.get('type')=='report-chunk' and x['id']==meta['id']),key=lambda x:x['index'])
    assert [x['index'] for x in chunks]==list(range(meta['chunks']))
    compressed=base64.b64decode(''.join(x['data'] for x in chunks),validate=True)
    assert hashlib.sha256(compressed).hexdigest()==meta['gzip_sha256']
    body=gzip.decompress(compressed)
    assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256']
    report=json.loads(body)
    (root/f"report-{meta['id']}.json.gz").write_bytes(compressed)
    reports[meta['id']]={'sha256':meta['sha256'],'gzip_sha256':meta['gzip_sha256'],'bytes':len(body),'requestedUrl':report['requestedUrl']}
receipt={'context':next(x for x in events if x['type']=='context'),'samples':[x for x in events if x['type']=='sample'],'result':[x for x in events if x['type']=='result'],'refused':[x for x in events if x['type']=='refused'],'reports':reports,'log_sha256':hashlib.sha256(raw).hexdigest()}
(root/'audit-readback.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/'audit.log.gz').write_bytes(gzip.compress(raw,mtime=0))
print(json.dumps({'context':receipt['context'],'sample_summaries':[{k:v for k,v in x.items() if k!='failedAudits'} for x in receipt['samples']],'reports_verified':len(reports),'result_count':len(receipt['result']),'refused':receipt['refused']},indent=2))
