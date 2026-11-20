"""Versioned single-secondary eval wrapper around frozen v5 worker20/engine20."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
V = HERE
BASE = V.parent.parent
WAVE = BASE / 'wave2-dev503-20261114'
ALLOWED = {'R_LOCAL_PROXY', 'R_FULL', 'R_SHUFFLED', 'MLP-C-FKL-I0', 'TR1-C-FKL-I0'}
sys.path.insert(0, str(V))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_bytes())


def write(path, value):
    import common
    common.write(path, value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def roster_by_id():
    frozen = read(WAVE/'freeze.json')
    roster = read(WAVE/'roster503.json')
    require(len(roster) == 503 and len({r['request_id'] for r in roster}) == 503, '503 roster invalid')
    require(sha(WAVE/'roster503.json') == frozen['roster_sha256'], '503 roster SHA changed')
    return {r['request_id']: r for r in roster}


def policy(group):
    require(group in ALLOWED, 'group outside secondary-only scope')
    import checkpoints as CP
    cfgpath = V/'configs'/(group+'.json')
    cfg = read(cfgpath)
    require(not cfg['evaluation_only'] and cfg['checkpoint_owner'] == group, 'checkpoint owner changed')
    train = V/'runs'/group/'train'
    selected = read(train/'selected.json')
    terminal = read(train/'completion.json')
    require(set(selected) == {'main', 'secondary'}, 'training choices changed')
    require(terminal['status'] == 'complete' and terminal['updates'] == cfg['updates'] and terminal['selected'] == selected, 'training terminal differs')
    cal = []
    for epoch in range(1, 6):
        path = train/f'calibration_epoch{epoch}.json'
        row = read(path)
        requests = row.get('requests', row.get('request_values'))
        require(row['epoch'] == epoch and len(requests) == 503, 'calibration coverage differs')
        value = row['calibration_request_equal_CE']
        require(isinstance(value, (float, int)) and value == value and abs(value) != float('inf'), 'invalid calibration CE')
        cal.append({'epoch': epoch, 'ce': value, 'sha256': sha(path)})
    best = min(cal, key=lambda x: (x['ce'], x['epoch']))
    require(selected['secondary']['epoch'] == best['epoch'], 'secondary is not minimum dev CE')
    binding, _ = CP.binding(V, group, 'secondary')
    require(binding['checkpoint_sha256'] == selected['secondary']['sha256'] and binding['epoch'] == best['epoch'], 'selected checkpoint differs')
    return {'group': group, 'formal_selection': 'secondary', 'selected_epoch': best['epoch'],
            'selected_ce': best['ce'], 'checkpoint_binding': binding,
            'selected_sha256': sha(train/'selected.json'), 'train_terminal_sha256': sha(train/'completion.json'),
            'calibration': cal, 'roster503_sha256': sha(WAVE/'roster503.json'),
            'runtime_freeze_sha256': sha(V/'runtime_freeze.json'),
            'eval_adapter_sha256': sha(__file__)}


def check_no_main(out):
    require(not (out/'main').exists(), 'unexpected main eval selection')


def check_receipt(receipt, dest, frozen, pol):
    rid = frozen['request_id']
    require(receipt.get('status') == 'pass' and receipt.get('request_id') == rid and receipt.get('selection') == 'secondary', 'sealed request identity/selection differs')
    require(receipt.get('checkpoint_binding') == pol['checkpoint_binding'], 'sealed checkpoint binding differs')
    require(receipt.get('checkpoint_sha256') == pol['checkpoint_binding']['checkpoint_sha256'], 'sealed checkpoint SHA differs')
    require(receipt.get('formal_policy_sha256') == digest(pol), 'sealed policy SHA differs')
    oracle_sha = frozen.get('oracle_receipt_sha256') or sha(WAVE/'oracles'/(rid+'.json'))
    require(receipt.get('oracle_receipt_sha256') == oracle_sha, 'oracle receipt SHA differs')
    require(receipt.get('raw_trace_sha256') == frozen['raw_trace_sha256'], 'RAW trace SHA differs')
    require(receipt.get('output_ids') == frozen['frozen_target_ids'] and receipt.get('eos') == frozen['eos'] and receipt.get('truncated') == frozen['truncated'], 'Target ID/EOS/truncation differs')
    for name, expected in receipt.get('files', {}).items():
        require(sha(dest/name) == expected, 'sealed file SHA differs: '+name)
    require(set(receipt.get('files', {})) == {'rounds.jsonl', 'signals.safetensors'}, 'sealed file set differs')
    require((dest/'rounds.jsonl').read_bytes().endswith(b'\n'), 'last round incomplete')
    return {'path': str(dest), 'receipt_sha256': sha(dest/'receipt.json')}


def scan(out, pol, roster, require_complete=False):
    out = pathlib.Path(out)
    check_no_main(out)
    if (out/'eval_policy.json').exists():
        require(read(out/'eval_policy.json') == pol, 'eval policy changed on resume')
    else:
        require(not require_complete, 'terminal missing eval policy')
    secondary = out/'secondary'
    if secondary.exists():
        require({p.name for p in secondary.iterdir() if p.is_dir()}.issubset(roster), 'extra secondary request directory')
    expected = {}
    for rid, frozen in roster.items():
        dest = secondary/rid
        if not dest.exists():
            continue
        require((dest/'receipt.json').is_file(), 'half-sealed request')
        expected['secondary/'+rid] = check_receipt(read(dest/'receipt.json'), dest, frozen, pol)
    old = read(out/'sealed_index.json') if (out/'sealed_index.json').exists() else {}
    require(set(old).issubset(expected), 'indexed request missing')
    for key in old:
        require(old[key] == expected[key], 'sealed index conflicts with request')
    if require_complete:
        require(len(expected) == 503 and set(expected) == {'secondary/'+rid for rid in roster}, '503 secondary requests incomplete')
        require(old == expected, 'terminal sealed index incomplete')
    return expected


def validate_terminal(group):
    pol = policy(group)
    out = V/'runs'/group/'eval'
    roster = roster_by_id()
    expected = scan(out, pol, roster, require_complete=True)
    term = read(out/'completion.json')
    require(term.get('status') == 'complete' and term.get('phase') == 'eval' and term.get('group') == group, 'eval terminal differs')
    require(term.get('formal_selection') == 'secondary' and term.get('formal_policy_sha256') == digest(pol) and term.get('sealed') == 503, 'secondary-only terminal differs')
    require(set(term.get('summaries', {})) == {'secondary'} and term['summaries']['secondary']['requests'] == 503, 'extra eval selection in terminal')
    require(term.get('selection_aliases') == [], 'selection alias in formal eval')
    require(term.get('sealed_index_sha256') == sha(out/'sealed_index.json'), 'terminal index SHA differs')
    require(term.get('roster_sha256') == pol['roster503_sha256'] and term.get('runtime_freeze_sha256') == pol['runtime_freeze_sha256'], 'terminal frozen binding differs')
    return 'complete'


def run(group):
    import common
    import engine20
    import worker20
    require(sha(__file__) == read(HERE/'secondary_eval_adapter_config.json')['source_sha256'], 'adapter source changed')
    common.verify_freeze()
    pol = policy(group)
    out = V/'runs'/group/'eval'
    out.mkdir(parents=True, exist_ok=True)
    roster = roster_by_id()
    scan(out, pol, roster)
    if (out/'completion.json').exists():
        validate_terminal(group)
        return
    if not (out/'eval_policy.json').exists():
        write(out/'eval_policy.json', pol)
    selected_path = (V/'runs'/group/'train/selected.json').resolve()
    original_read = common.read
    original_write = common.write

    def guarded_read(path):
        value = original_read(path)
        p = pathlib.Path(path).resolve()
        if p == selected_path:
            require(set(value) == {'main', 'secondary'}, 'training selection set differs')
            return {'secondary': value['secondary']}
        if p.name == 'receipt.json' and p.is_relative_to(out/'secondary'):
            rid = p.parent.name
            require(rid in roster, 'extra request receipt')
            check_receipt(value, p.parent, roster[rid], pol)
        return value

    def guarded_write(path, value):
        p = pathlib.Path(path)
        if p.name == 'receipt.json' and p.is_relative_to(out/'incomplete'):
            require(value.get('selection') == 'secondary', 'attempted main request write')
            rid = value['request_id']
            require(rid in roster and value.get('checkpoint_binding') == pol['checkpoint_binding'], 'request binding differs')
            value = dict(value, checkpoint_sha256=pol['checkpoint_binding']['checkpoint_sha256'],
                         oracle_receipt_sha256=sha(WAVE/'oracles'/(rid+'.json')),
                         raw_trace_sha256=roster[rid]['raw_trace_sha256'],
                         formal_policy_sha256=digest(pol))
        if p == out/'completion.json':
            require(set(value.get('summaries', {})) == {'secondary'} and value['summaries']['secondary']['requests'] == 503, 'attempted dual eval terminal')
            require(value.get('selection_aliases') == [], 'attempted alias eval terminal')
            value = dict(value, formal_selection='secondary', formal_policy_sha256=digest(pol), sealed=503)
        original_write(path, value)

    engine20.read = guarded_read
    engine20.write = guarded_write
    worker20.read = guarded_read
    sys.argv = [str(V/'worker20.py'), '--group', group, '--phase', 'eval']
    worker20.main()
    validate_terminal(group)


def fixtures():
    tests = []
    with tempfile.TemporaryDirectory(prefix='task59-secondary-eval-') as tmp:
        out = pathlib.Path(tmp)/'eval'
        out.mkdir()
        (out/'main').mkdir()
        try: check_no_main(out)
        except RuntimeError: tests.append('extra_main_rejected')
        else: raise AssertionError('main accepted')
        (out/'main').rmdir()
        train = pathlib.Path(tmp)/'selected.json'
        train.write_text('{"main":{"sha256":"a"},"secondary":{"sha256":"b"}}')
        choices = read(train)
        require(set(choices) == {'main','secondary'} and list({'secondary':choices['secondary']}) == ['secondary'], 'selection filter failed')
        tests.append('secondary_only_filter')
        fake = {'request_id':'req','raw_trace_sha256':'raw','oracle_receipt_sha256':'good',
                'frozen_target_ids':[1], 'eos':False,'truncated':True}
        pol = {'checkpoint_binding':{'checkpoint_sha256':'cp'}}
        dest = out/'secondary'/'req'
        dest.mkdir(parents=True)
        (dest/'rounds.jsonl').write_text('{"round":0}\n')
        (dest/'signals.safetensors').write_bytes(b'x')
        receipt = {'status':'pass','request_id':'req','selection':'secondary','checkpoint_binding':pol['checkpoint_binding'],
                   'checkpoint_sha256':'cp','formal_policy_sha256':digest(pol),'oracle_receipt_sha256':'wrong',
                   'raw_trace_sha256':'raw','output_ids':[1],'eos':False,'truncated':True,
                   'files':{'rounds.jsonl':sha(dest/'rounds.jsonl'),'signals.safetensors':sha(dest/'signals.safetensors')}}
        write(dest/'receipt.json', receipt)
        try: check_receipt(receipt,dest,fake,pol)
        except RuntimeError: tests.append('wrong_oracle_sha_rejected')
        else: raise AssertionError('wrong oracle accepted')
        receipt['oracle_receipt_sha256']='good'
        write(dest/'receipt.json',receipt)
        first=scan(out,pol,{'req':fake})
        require(first==scan(out,pol,{'req':fake}), 'orphaned seal not adopted idempotently')
        tests.append('orphaned_seal_double_resume')
        write(out/'sealed_index.json',{'secondary/req':{'receipt_sha256':'wrong'}})
        try: scan(out,pol,{'req':fake})
        except RuntimeError: tests.append('conflicting_index_rejected')
        else: raise AssertionError('conflicting index accepted')
    return tests


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--group', choices=sorted(ALLOWED))
    parser.add_argument('--selftest', action='store_true')
    a=parser.parse_args()
    if a.selftest:
        print(json.dumps({'status':'pass','tests':fixtures(),'source_sha256':sha(__file__)}))
    else:
        require(a.group is not None, '--group required')
        run(a.group)
