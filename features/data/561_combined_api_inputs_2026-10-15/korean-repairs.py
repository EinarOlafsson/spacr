from pathlib import Path
import hashlib,json,sys
root=Path.cwd();scratch=Path(__file__).parent;sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as a;import build_guide_i18n as g
before=json.loads((scratch/'preflight.json').read_text());rows={r['label']:r for r in before['issues']};sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
targets={
4:'이 값은 :func:`torch.load` 함수에 전달됩니다. 기본값 ``cpu`` 를 사용하면 GPU 에서 학습한 체크포인트를 GPU 가 없는 컴퓨터에서도 불러올 수 있습니다. 인식할 수 없는 장치 문자열이면 torch 가 ``RuntimeError`` 예외를 발생시킵니다.',
5:'``None`` 을 지정하면 기록된 구성에서 아키텍처를 재구성합니다. 기존의 상태 키-값 매핑만 있는 파일에는 아키텍처가 기록되지 않으므로 ``maxvit_t`` 를 알림 없이 가정합니다. 여기에 모듈을 전달하면 해당 모듈에 상태를 불러오고 동일한 객체를 반환합니다. 기존 전체 모듈 파일에서는 이 인수를 무시하고 파일에 저장된 모듈을 반환합니다.',
6:'``load_state_dict`` 에 전달됩니다. ``True`` 는 키가 하나라도 일치하지 않으면 ``RuntimeError`` 예외를 발생시킵니다. ``False`` 는 누락되거나 예상하지 못한 키를 허용하므로, 아키텍처가 맞지 않아도 일부가 여전히 무작위로 초기화된 채 별도 알림 없이 불러옵니다. 기존 전체 모듈 파일에서는 이 인수도 무시됩니다.',
7:'파일이 모듈도 체크포인트 매핑도 아니거나, ``artifact_version`` 값이 지원되는 버전이 아니거나, 상태 키-값 매핑을 찾지 못했거나, 구성에 아키텍처 이름이 없고 ``model`` 도 제공되지 않은 경우입니다.'}
repairs=[]
for i,target in targets.items():
 label=f'spacr.torch_artifacts.load_model_artifact#{i}';row=rows[label];source=row['source'];context=row['context']
 assert a._reviewed_api_block_valid(source,target,'ko') and a._reviewed_api_block_valid(context,target,'ko')
 assert not list(g.inline_markup_problems(target))
 repairs.append({'language':'ko','label':label,'old_label':f'spacr.torch_artifacts.load_model_artifact#{i-1}','source':source,'source_sha256':sha(source),'context':context,'context_sha256':sha(context),'old_target':row['target'],'old_target_sha256':sha(row['target']),'new_target':target,'new_target_sha256':sha(target),'normal_source_context_markup':'PASS','reason':'Restore actual RST literal/role boundaries and exact loading/legacy semantics; preserves old catalog bytes as evidence.'})
(scratch/'korean-repairs.json').write_text(json.dumps(repairs,ensure_ascii=False,indent=2)+'\n');print('Four exact Korean source/context/inline checks PASS; independent semantic peer pending')
