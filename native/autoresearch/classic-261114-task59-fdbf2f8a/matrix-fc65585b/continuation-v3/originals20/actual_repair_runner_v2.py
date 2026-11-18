"""Task34 real-model RAW and three matched-capacity repair arms.

The CPU dependency gate runs before CUDA/model loading. This file is deliberately
independent of the old task19 scheduler and does not dispatch itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from actual_cpu_entry_v1 import ROOT, TASK28, sha, read, save, need
from task28_terminal_gate_v2 import actual_task28_gate

sys.path[:0] = [str(ROOT/'frozen_v96/src'), str(TASK28)]
from source_snapshot import base_reader as b
from feedback_codec.v10_four_arm import V10FourArmHead
from teacher_loop import RoundContext, TeacherMessage, TeacherRepairHead
from repair_reference import weighted_loss
import torch

SEED = 42
EOS = 151645
N_TRAIN = 7982
N_DEV = 1598
ARMS = {'R_LOCAL_PROXY':'local_proxy', 'R_FULL':'matched', 'R_SHUFFLED':'shuffled'}


def physical_gpu(index):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True).splitlines()
    gpu={int(x[0]):x for x in ([z.strip() for z in row.split(',')] for row in rows)}[index]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                  '--format=csv,noheader,nounits'],text=True).splitlines()
    return {'index':index,'uuid':gpu[1],'free_mib':int(gpu[2]),
            'utilization_percent':int(gpu[3]),
            'compute_pids':[row.strip() for row in apps if gpu[1] in row]}


def eligible(gpu):
    return gpu['free_mib']>=22000 and gpu['utilization_percent']<=10 and not gpu['compute_pids']


def verify_worker_admission(stage):
    """Inherited cross-root UUID lease plus second strict admission before CUDA."""
    index=int(os.environ['TASK34_GPU_INDEX'])
    uuid=os.environ['TASK34_GPU_UUID']
    fd=int(os.environ['TASK34_LEASE_FD'])
    lease=Path(os.readlink(f'/proc/self/fd/{fd}'))
    need(lease==ROOT.parent/'gpu_claims_v1'/(uuid+'.lock'), 'lease path differs')
    claim=read(lease)
    need(claim['stage']==stage and claim['index']==index and claim['uuid']==uuid and
         claim['pid']==os.getppid(), 'lease owner/stage differs')
    mount=subprocess.check_output(['findmnt','-T',str(ROOT),'-n','-o','OPTIONS'],text=True)
    need('rw' in mount.strip().split(','), 'task34 root mount not writable')
    first=physical_gpu(index)
    time.sleep(2)
    second=physical_gpu(index)
    receipt={'stage':stage,'worker_pid':os.getpid(),'parent_pid':os.getppid(),
             'gpu_uuid':uuid,'admission_first':first,'admission_second':second,
             'lease_path':str(lease),'worker_source_sha256':sha(Path(__file__)),
             'checked_unix_s':time.time()}
    path=ROOT/'admission_v2'/(stage+'.json')
    save(path,receipt)
    need(first['uuid']==second['uuid']==uuid and eligible(first) and eligible(second),
         'worker double GPU admission failed')


def tensor_digest(values):
    return b.state_digest(values)


def actual_model_digest():
    return read(ROOT/'actual_depmanifest_v1.json')['target_model_manifest_digest_sha256']


def assert_gate():
    complete, detail, _ = actual_task28_gate()
    need(complete, 'task28 four-arm real terminal is required before new GPU work: '+str(detail))
    receipt = read(ROOT/'actual_cpu_entry_v1.json')
    deps = read(ROOT/'actual_depmanifest_v1.json')
    need(receipt['state'] == 'pass' and receipt['server_real_locked_data'] and
         receipt['dependency_manifest_sha256'] == sha(ROOT/'actual_depmanifest_v1.json') and
         deps['cpu_entry_source_sha256'] == sha(ROOT/'actual_cpu_entry_v1.py'),
         'actual task34 CPU/dependency receipt differs')
    need(deps['frozen_manifest_sha256'] == sha(ROOT/'frozen_v96/manifest.json') and
         deps['frozen_contract_sha256'] == sha(ROOT/'frozen_v96/contract.json'),
         'frozen v96 inputs differ')


def context(window, split):
    o = window.observation
    need(o.target_hidden_prefix == o.valid_prefix and
         o.corrected_prefix[:-1] == o.valid_prefix,
         'Target hidden is not the legal previous valid-prefix boundary')
    return RoundContext(o.prompt_id, o.family_id, 'dev' if split == 'val' else 'train',
                        o.round_index, tuple(o.valid_prefix), tuple(o.corrected_prefix),
                        actual_model_digest())


def pairing(windows):
    """Deterministic same-split different-family permutation of window indices."""
    order = sorted(range(len(windows)), key=lambda i:(windows[i].observation.family_id,
                                                      windows[i].observation.prompt_id,
                                                      windows[i].observation.round_index))
    counts = defaultdict(int)
    for i in order:
        counts[windows[i].observation.family_id] += 1
    maximum = max(counts.values())
    shift = maximum + random.Random(SEED).randrange(maximum)
    need(shift * 2 <= len(order), 'no legal family derangement')
    donors = {}
    for pos, i in enumerate(order):
        donor = order[(pos+shift) % len(order)]
        need(windows[i].observation.family_id != windows[donor].observation.family_id,
             'family derangement failed')
        donors[i] = donor
    need(set(donors.values()) == set(range(len(order))), 'donor is not a permutation')
    return donors


def pairing_manifest(train_windows, dev_windows):
    path = ROOT/'pairing_manifest_v2.json'
    if path.is_file():
        obj = read(path)
    else:
        obj = {'scope':'task34_fixed_different_family_pairing_v2', 'seed':SEED,
               'splits':{}}
        for split, windows in (('train',train_windows),('val',dev_windows)):
            donors = pairing(windows)
            obj['splits'][split] = [
                {'destination_window_id':b.previous.causal_window_id(windows[i]),
                 'source_window_id':b.previous.causal_window_id(windows[donors[i]]),
                 'destination_family_id':windows[i].observation.family_id,
                 'source_family_id':windows[donors[i]].observation.family_id}
                for i in range(len(windows))]
        save(path,obj)
    for split, windows in (('train',train_windows),('val',dev_windows)):
        rows=obj['splits'][split]
        need(len(rows)==len(windows),'pairing length differs')
        ids={b.previous.causal_window_id(w):i for i,w in enumerate(windows)}
        need(len(ids)==len(windows),'duplicate causal window ID')
        donors=[]
        for i,row in enumerate(rows):
            need(row['destination_window_id']==b.previous.causal_window_id(windows[i]),
                 'pairing destination differs')
            donor=ids[row['source_window_id']]
            need(windows[donor].observation.family_id!=windows[i].observation.family_id,
                 'pairing donor family differs')
            donors.append(donor)
        need(set(donors)==set(range(len(windows))),'pairing ceased to be a permutation')
    mapping={}
    for split,windows in (('train',train_windows),('val',dev_windows)):
        ids={b.previous.causal_window_id(w):i for i,w in enumerate(windows)}
        mapping[split]=[ids[row['source_window_id']] for row in obj['splits'][split]]
    return sha(path),mapping


def message(window, split, mode, donor=None, pair_sha=None):
    c = context(window,split)
    if mode == 'local_proxy':
        return c,None
    if mode == 'matched':
        m = TeacherMessage.full_hidden(window.observation.target_hidden.unsqueeze(0),c)
    elif mode == 'shuffled':
        need(donor is not None and pair_sha is not None,'shuffled donor lock missing')
        source=context(donor,split)
        m=TeacherMessage.full_hidden(donor.observation.target_hidden.unsqueeze(0),
                                     source,c,pairing_lock_sha256=pair_sha)
    else:
        raise ValueError('unknown repair mode')
    m.validate(c,shuffled=mode=='shuffled',feature_dim=4096)
    need(len(m.payload)==8192,'full teacher payload must be BF16 8192B')
    return c,m


def init_models():
    target,draft,draft_model,config=b.model_loader._load_models(
        SimpleNamespace(target=Path('/mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B'),
                        draft=Path('/mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B')),
        EOS)
    need((config.window,config.rank,config.groups,config.choices,config.code_dim,
          config.vocab_size,config.target_hidden_size,config.draft_hidden_size)
         == (8,32,8,256,16,151936,4096,1024),'locked model dimensions differ')
    del target
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base=V10FourArmHead(config,draft_model.get_output_embeddings(),mode='C',
                        draft_embedding=draft_model.get_input_embeddings())
    head=TeacherRepairHead(base,'mlp','matched').to('cuda:0')
    return draft,head


def init_arm(draft_model,config,mode):
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base=V10FourArmHead(config,draft_model.get_output_embeddings(),mode='C',
                        draft_embedding=draft_model.get_input_embeddings())
    head=TeacherRepairHead(base,'mlp',mode).to('cuda:0')
    budget=head.parameter_budget()
    need(budget['unique_trainable_total']==7914403 and
         budget['teacher_bridge_trainable']==270528,
         'new equal-capacity MLP parameter budget differs')
    return head


def free(head,draft,state,c,m):
    prefix=list(state.corrected_prefix)
    out=[]
    for pos in range(1,9):
        token=head.next_token(prefix,state,pos,draft,context=c,message=m)
        out.append(token)
        prefix.append(token)
        if token==EOS:
            break
    return out


def raw():
    assert_gate()
    verify_worker_admission('task34-RAW200-v1')
    out=ROOT/'raw200_v1'
    out.mkdir(exist_ok=True)
    status={'state':'running','phase':'load_val','pid':os.getpid(),'prompts':0,
            'windows':0,'candidate_token_mismatches':0,'prior_RAW_mismatches':0}
    save(out/'status.json',status)
    start=time.monotonic()
    try:
        _,prompts,windows=b.load_split('val')
        by_prompt=defaultdict(list)
        for w in windows:by_prompt[w.observation.prompt_id].append(w)
        target,draft,draft_model,config=b.model_loader._load_models(
            SimpleNamespace(target=Path('/mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B'),
                            draft=Path('/mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B')),EOS)
        need((config.window,config.vocab_size,config.target_hidden_size,config.draft_hidden_size)
             == (8,151936,4096,1024),'RAW model dimensions differ')
        del target
        torch.cuda.empty_cache()
        head=init_arm(draft_model,config,'matched')
        initial=tensor_digest(head.snapshot_state())
        initial_path=out/'initial_snapshot.pt'
        b.save(initial_path,head.snapshot_state())
        save(out/'initial_manifest.json',{'seed':SEED,'initial_state_sha256':initial,
             'snapshot_file_sha256':sha(initial_path),'parameter_budget':head.parameter_budget(),
             'depmanifest_sha256':sha(ROOT/'actual_depmanifest_v1.json')})
        status['phase']='score_initial_RAW'
        save(out/'status.json',status)
        head.eval()
        with torch.inference_mode():
            for p in prompts:
                rows=[]
                prior_path=TASK28/'stage0_v1/scores'/(
                    hashlib.sha256(p.id.encode()).hexdigest()+'.json')
                prior_rows=read(prior_path)['rows']
                prior_by_id={r['window_id']:r for r in prior_rows}
                need(len(prior_by_id)==len(by_prompt[p.id]),'prior RAW window count differs')
                for w in by_prompt[p.id]:
                    c,m=message(w,'val','matched')
                    state=head.prepare_observation(w.observation,draft)
                    new=free(head,draft,state,c,m)
                    original=b.previous._raw_free(draft,w.observation,EOS)
                    legacy=b.previous._score_row(SEED,'TASK34-INITIAL-RAW',w,new,EOS)
                    legacy['raw_proposed_ids']=original
                    prior=prior_by_id[legacy['window_id']]['proposed_ids']
                    legacy['prior_locked_RAW_proposed_ids']=prior
                    legacy['teacher_payload_bytes']=len(m.payload)
                    legacy['candidate_token_equal']=new==original
                    legacy['prior_RAW_equal']=new==prior
                    legacy['eos_equal']=(EOS in new)==(EOS in original)
                    legacy['first_token_mismatch_position']=next(
                        (i for i,(x,y) in enumerate(zip(new,original),1) if x!=y),
                        None if len(new)==len(original) else min(len(new),len(original))+1)
                    status['candidate_token_mismatches']+=int(new!=original)
                    status['prior_RAW_mismatches']+=int(new!=prior)
                    rows.append(legacy)
                save(out/'scores'/(hashlib.sha256(p.id.encode()).hexdigest()+'.json'),
                     {'prompt_id':p.id,'family_id':p.family_id,'source':p.task_type,'rows':rows})
                status['prompts']+=1
                status['windows']+=len(rows)
                if status['prompts']%10==0:save(out/'status.json',status)
        need(status['prompts']==200 and status['windows']==1598,
             'initial RAW coverage differs')
        need(status['candidate_token_mismatches']==0 and status['prior_RAW_mismatches']==0,
             'initial RAW candidate identity failed')
        gate={'state':'pass','scope':'task34_new_initial_RAW_200_v1','prompts':200,
              'legal_windows':1598,'candidate_token_mismatches':0,'prior_RAW_mismatches':0,
              'initial_state_sha256':initial,'initial_manifest_sha256':sha(out/'initial_manifest.json'),
              'score_file_sha256':{p.name:sha(p) for p in sorted((out/'scores').glob('*.json'))},
              'runner_sha256':sha(Path(__file__))}
        save(out/'raw_gate_v1.json',gate)
        status.update(state='complete',phase='complete')
    except BaseException as exc:
        status.update(state='failed',error_type=type(exc).__name__,error=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        status['elapsed_gpu_seconds']=time.monotonic()-start
        save(out/'status.json',status)
        save(out/'exit_receipt.json',{'state':status['state'],'status_sha256':sha(out/'status.json'),
                                     'raw_gate_sha256':sha(out/'raw_gate_v1.json') if
                                     (out/'raw_gate_v1.json').is_file() else None})


def train_step(head,draft,w,c,m,optimizer,step_seed):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    state=head.prepare_observation(w.observation,draft)
    need(state.target_hidden is None and state.target_top32_features is None,
         'teacher feature entered ordinary C state')
    with torch.no_grad():
        hidden=draft.teacher_hidden(list(state.corrected_prefix),list(w.labels))
    torch.manual_seed(step_seed)
    logits=head.forward_logits(hidden.detach().unsqueeze(0),state,context=c,message=m)
    need(logits.shape==(1,len(w.labels),151936),'repair logits shape differs')
    valid=torch.ones((1,len(w.labels)),device=logits.device,dtype=torch.bool)
    labels=torch.tensor([list(w.labels)],device=logits.device,dtype=torch.long)
    loss=weighted_loss(logits,valid,labels=labels)
    need(bool(torch.isfinite(loss)),'repair loss nonfinite')
    loss.backward()
    grads=[p.grad for p in head.parameters() if p.requires_grad and p.grad is not None]
    need(grads and all(bool(torch.isfinite(g).all()) for g in grads),
         'repair gradient missing or nonfinite')
    optimizer.step()
    return float(loss.detach().item()),len(w.labels)


def checkpoint(path,head,optimizer,lock_sha,initial_sha,epoch,position,updates,history,
               loss_sum,token_count):
    b.save(path,{'model':head.snapshot_state(),'optimizer':optimizer.state_dict(),
                 'train_lock_sha256':lock_sha,'initial_sha256':initial_sha,
                 'epoch':epoch,'position':position,'updates':updates,
                 'history':history,'loss_sum':loss_sum,'token_count':token_count,
                 'torch_cpu_rng':torch.get_rng_state(),
                 'torch_cuda_rng':torch.cuda.get_rng_state_all(),
                 'python_rng':random.getstate()})


def score_arm(arm,head,draft,prompts,windows,split,donors,worker,status,pair_sha):
    head.eval()
    by_prompt=defaultdict(list)
    by_index={id(w):i for i,w in enumerate(windows)}
    for w in windows:by_prompt[w.observation.prompt_id].append(w)
    scores=worker/(split+'_scores')
    scores.mkdir(exist_ok=True)
    prompt_means=[]
    main_means=[]
    legal_windows=main_windows=0
    for prompt in prompts:
        path=scores/(hashlib.sha256(prompt.id.encode()).hexdigest()+'.json')
        if path.exists():
            obj=read(path)
            need(obj['prompt_id']==prompt.id and obj['arm']==arm,'score receipt differs')
        else:
            rows=[]
            with torch.inference_mode():
                for w in by_prompt[prompt.id]:
                    donor=windows[donors[by_index[id(w)]]] if arm=='R_SHUFFLED' else None
                    c,m=message(w,split,ARMS[arm],donor,pair_sha)
                    state=head.prepare_observation(w.observation,draft)
                    proposed=free(head,draft,state,c,m)
                    row=b.previous._score_row(SEED,arm,w,proposed,EOS)
                    row.update(teacher_payload_bytes=0 if m is None else len(m.payload),
                               teacher_source_window_id=None if donor is None else
                               b.previous.causal_window_id(donor),
                               teacher_context_round_key=list(c.round_key),
                               eos_reached=bool(proposed and proposed[-1]==EOS))
                    rows.append(row)
            obj={'arm':arm,'split':split,'prompt_id':prompt.id,
                 'family_id':prompt.family_id,'source':prompt.task_type,'rows':rows}
            save(path,obj)
        need(len(obj['rows'])==len(by_prompt[prompt.id]),'score windows differ')
        vals=[r['acceptance'] for r in obj['rows']]
        prompt_means.append(sum(vals)/len(vals))
        main=[r['acceptance'] for r in obj['rows'] if r['full_window_eligible']]
        if main:main_means.append(sum(main)/len(main))
        legal_windows+=len(vals)
        main_windows+=len(main)
        status[split+'_prompts_scored']=len(prompt_means)
        if len(prompt_means)%10==0:save(worker/'status.json',status)
    need(legal_windows==(N_TRAIN if split=='train' else N_DEV),'score coverage differs')
    if split=='val':need(main_windows==1596,'dev main coverage differs')
    summary={'arm':arm,'split':split,'prompts':len(prompts),'legal_windows':legal_windows,
             'main_windows':main_windows,'prompt_equal_legal_mean':sum(prompt_means)/len(prompt_means),
             'prompt_equal_main_mean':sum(main_means)/len(main_means),
             'score_file_sha256':{p.name:sha(p) for p in sorted(scores.glob('*.json'))}}
    save(worker/(split+'_summary.json'),summary)
    return summary


def repair(arm):
    assert_gate()
    verify_worker_admission('task34-'+arm+'-v1')
    raw_gate=read(ROOT/'raw200_v1/raw_gate_v1.json')
    need(raw_gate['state']=='pass' and raw_gate['prompts']==200 and
         raw_gate['legal_windows']==1598 and raw_gate['candidate_token_mismatches']==0,
         'new full200 initial RAW gate missing or failed')
    mode=ARMS[arm]
    worker=ROOT/'repairs_v1'/arm
    worker.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    status={'state':'running','phase':'load_data','arm':arm,'pid':os.getpid(),
            'training_updates':0}
    save(worker/'status.json',status)
    try:
        _,tp,tw=b.load_split('train')
        _,vp,vw=b.load_split('val')
        pair_sha,donors=pairing_manifest(tw,vw)
        anchor=read(TASK28/'handoff/anchors/anchor_A_lock.json')
        order=anchor['epoch_order_indices']
        need(len(order)==5 and all(len(x)==N_TRAIN for x in order),
             'locked training order differs')
        lock={'scope':'task34_repair_front_ce_v1','arm':arm,'signal_mode':mode,
              'seed':SEED,'epochs':5,'train_windows':N_TRAIN,'dev_windows':N_DEV,
              'task28_anchor_sha256':sha(TASK28/'handoff/anchors/anchor_A_lock.json'),
              'task28_train_acceptance_sha256':sha(TASK28/'train_acceptance.json'),
              'task28_val_acceptance_sha256':sha(TASK28/'val_acceptance.json'),
              'task34_raw_gate_sha256':sha(ROOT/'raw200_v1/raw_gate_v1.json'),
              'pairing_manifest_sha256':pair_sha,
              'depmanifest_sha256':sha(ROOT/'actual_depmanifest_v1.json'),
              'runner_sha256':sha(Path(__file__)),
              'frozen_teacher_loop_sha256':sha(ROOT/'frozen_v96/src/teacher_loop.py'),
              'optimizer':{'kind':'AdamW','lr':1e-4,'betas':[.9,.999],
                           'eps':1e-8,'weight_decay':.01},
              'loss':'position8to1_normalized_CE','final_checkpoint':'epoch5',
              'epoch_order_indices':order,
              'window_ids':[b.previous.causal_window_id(w) for w in tw],
              'label_lengths':[len(w.labels) for w in tw]}
        lock_path=worker/'train_lock.json'
        if lock_path.exists():need(read(lock_path)==lock,'existing repair lock differs')
        else:save(lock_path,lock)
        lock_sha=sha(lock_path)
        status['phase']='load_models'
        save(worker/'status.json',status)
        target,draft,draft_model,config=b.model_loader._load_models(
            SimpleNamespace(target=Path('/mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B'),
                            draft=Path('/mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B')),EOS)
        need((config.window,config.vocab_size,config.target_hidden_size,config.draft_hidden_size)
             == (8,151936,4096,1024),'model dimensions differ')
        del target
        torch.cuda.empty_cache()
        head=init_arm(draft_model,config,mode)
        initial=tensor_digest(head.snapshot_state())
        need(initial==raw_gate['initial_state_sha256'],'three arms must share new RAW initial state')
        save(worker/'initial_manifest.json',{'arm':arm,'initial_state_sha256':initial,
             'parameter_budget':head.parameter_budget(),'train_lock_sha256':lock_sha,
             'raw_gate_sha256':sha(ROOT/'raw200_v1/raw_gate_v1.json')})
        optimizer=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                                     lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        current=worker/'checkpoints/last.pt'
        epoch=position=updates=0
        history=[]
        loss_sum=0.0
        token_count=0
        if current.exists():
            old=torch.load(current,map_location='cpu',weights_only=False)
            need(old['train_lock_sha256']==lock_sha and old['initial_sha256']==initial,
                 'resume lock/initial differs')
            head.restore_snapshot(old['model'])
            optimizer.load_state_dict(old['optimizer'])
            epoch,position,updates=old['epoch'],old['position'],old['updates']
            history,loss_sum,token_count=old['history'],old['loss_sum'],old['token_count']
            torch.set_rng_state(old['torch_cpu_rng'])
            torch.cuda.set_rng_state_all(old['torch_cuda_rng'])
            random.setstate(old['python_rng'])
        else:
            checkpoint(current,head,optimizer,lock_sha,initial,epoch,position,updates,
                       history,loss_sum,token_count)
        status.update(phase='train',epoch=epoch,position=position,training_updates=updates,
                      initial_state_sha256=initial)
        save(worker/'status.json',status)
        while epoch<5:
            while position<N_TRAIN:
                i=order[epoch][position]
                w=tw[i]
                donor=tw[donors['train'][i]] if arm=='R_SHUFFLED' else None
                c,m=message(w,'train',mode,donor,pair_sha)
                loss,tokens=train_step(head,draft,w,c,m,optimizer,
                                       SEED*1_000_000+epoch*N_TRAIN+position+1)
                loss_sum+=loss
                token_count+=tokens
                position+=1
                updates+=1
                if position%256==0:
                    checkpoint(current,head,optimizer,lock_sha,initial,epoch,position,
                               updates,history,loss_sum,token_count)
                    status.update(epoch=epoch,position=position,training_updates=updates)
                    save(worker/'status.json',status)
            history.append({'epoch':epoch+1,'steps':N_TRAIN,
                            'window_mean_loss':loss_sum/N_TRAIN,'label_tokens':token_count})
            epoch+=1
            position=0
            loss_sum=0.0
            token_count=0
            checkpoint(worker/f'checkpoints/epoch{epoch}.pt',head,optimizer,
                       lock_sha,initial,epoch,position,updates,history,loss_sum,token_count)
            checkpoint(current,head,optimizer,lock_sha,initial,epoch,position,
                       updates,history,loss_sum,token_count)
            status.update(epoch=epoch,position=position,training_updates=updates)
            save(worker/'status.json',status)
        need(updates==39910 and len(history)==5,'five epochs incomplete')
        receipt={'arm':arm,'state':'trained','seed':SEED,'epochs':5,'completed_updates':updates,
                 'train_lock_sha256':lock_sha,'initial_state_sha256':initial,
                 'final_trainable_sha256':tensor_digest(head.snapshot_state()),
                 'epoch5_checkpoint_sha256':sha(worker/'checkpoints/epoch5.pt'),
                 'history':history,'parameter_budget':head.parameter_budget()}
        save(worker/'training_receipt.json',receipt)
        status['phase']='score_val'
        save(worker/'status.json',status)
        val=score_arm(arm,head,draft,vp,vw,'val',donors['val'],worker,status,pair_sha)
        status['phase']='score_train'
        save(worker/'status.json',status)
        train=score_arm(arm,head,draft,tp,tw,'train',donors['train'],worker,status,pair_sha)
        save(worker/'summary.json',{'arm':arm,'train':train,'val':val,
             'training_receipt_sha256':sha(worker/'training_receipt.json'),
             'elapsed_gpu_seconds':time.monotonic()-start})
        status.update(state='complete',phase='complete')
    except BaseException as exc:
        status.update(state='failed',error_type=type(exc).__name__,error=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        status['elapsed_gpu_seconds']=time.monotonic()-start
        save(worker/'status.json',status)
        save(worker/'exit_receipt.json',{'arm':arm,'state':status['state'],
             'status_sha256':sha(worker/'status.json'),
             'elapsed_gpu_seconds':status['elapsed_gpu_seconds']})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('raw','repair'))
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.phase=='raw':
        need(args.arm is None,'RAW does not take an arm')
        raw()
    else:
        need(args.arm is not None,'repair arm is required')
        repair(args.arm)
