"""Task59 canonical greedy SD loop. No oracle tokens enter proposal or feedback."""
import hashlib
import json
import time

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def decode(prompt_ids, *, draft_token, target_step, prepare, on_round,
           oracle_ids, max_new=128, window=8, eos=151645):
    confirmed=list(prompt_ids); generated=[]; state=None; rows=[]
    while len(generated)<max_new:
        started=time.perf_counter();before=confirmed.copy();proposal=[]
        draft_started=time.perf_counter()
        for position in range(1,min(window,max_new-len(generated))+1):
            token=int(draft_token(confirmed+proposal,state,position))
            proposal.append(token)
            if token==eos: break
        draft_seconds=time.perf_counter()-draft_started
        accepted=0;correction=None;bonus=None;rejected=None;boundary=None;decision_ids=[]
        verify_started=time.perf_counter()
        for token in proposal:
            step=target_step(confirmed.copy());decision_ids.append(int(step.token_id))
            if token!=int(step.token_id):
                rejected=token;correction=int(step.token_id);boundary=step
                confirmed.append(correction);generated.append(correction);break
            accepted+=1;confirmed.append(token);generated.append(token)
            if token==eos:break
        terminal=(generated[-1]==eos or len(generated)==max_new)
        if rejected is None and not terminal:
            boundary=target_step(confirmed.copy());bonus=int(boundary.token_id)
            decision_ids.append(bonus);confirmed.append(bonus);generated.append(bonus)
            terminal=(bonus==eos or len(generated)==max_new)
        verify_seconds=time.perf_counter()-verify_started
        answer=correction if correction is not None else bonus
        committed=proposal[:accepted]+([] if answer is None else [answer])
        assert committed and confirmed==before+committed
        valid=before+proposal[:accepted]
        row=dict(round=len(rows),prefix_before_ids=before,prefix_before_sha256=digest(before),
                 candidate_ids=proposal,target_decision_ids=decision_ids,accepted_count=accepted,
                 rejected_id=rejected,correction_id=correction,bonus_id=bonus,committed_ids=committed,
                 prefix_after_ids=confirmed.copy(),prefix_after_sha256=digest(confirmed),
                 valid_teacher_prefix_sha256=digest(valid) if answer is not None else None,
                 valid_teacher_prefix_length=len(valid) if answer is not None else None,
                 output_tokens_so_far=len(generated),feedback_from_round=len(rows)-1 if state is not None else None,
                 eos=generated[-1]==eos,truncated=len(generated)==max_new and generated[-1]!=eos,
                 draft_seconds=draft_seconds,verification_seconds=verify_seconds)
        # The oracle is an assertion only, after the actual Target verification.
        row['confirmed_prefix_matches_frozen_oracle']=generated==list(oracle_ids[:len(generated)])
        feedback_started=time.perf_counter()
        if not terminal and prepare is not None:
            assert boundary is not None and answer is not None
            state=prepare(row,boundary)
        row['feedback_prepare_seconds']=time.perf_counter()-feedback_started
        row['round_seconds']=time.perf_counter()-started
        on_round(row,boundary);rows.append(row)
        if not row['confirmed_prefix_matches_frozen_oracle']:
            raise RuntimeError('Target output diverged from frozen oracle at round '+str(row['round']))
        if terminal:break
    assert generated==list(oracle_ids)
    return dict(output_ids=generated,eos=bool(generated and generated[-1]==eos),
                truncated=len(generated)==max_new and generated[-1]!=eos,
                rounds=rows,accepted_draft_tokens=sum(r['accepted_count'] for r in rows),
                verification_rounds=len(rows),actual_target_forwards=sum(len(r['target_decision_ids']) for r in rows))

def cpu_tests():
    from types import SimpleNamespace
    cases=[]
    for name,seq,mode,eos in [
        ('all_accept_then_bonus_and_budget',list(range(1,21)),'match',99),
        ('zero_accept_every_round',list(range(1,10)),'reject',99),
        ('reject_after_two',list(range(1,19)),'reject_third',99),
        ('accepted_eos',[1,2,99],'match',99),
        ('correction_eos',[1,2,99],'reject',99),
        ('bonus_eos',list(range(1,9))+[99],'match',99),
        ('short_final_budget',[1,2,3],'match',99),
    ]:
        prompt=[900];events=[];prepared=[]
        def target(prefix):
            assert prefix==prompt+seq[:len(prefix)-1]
            return SimpleNamespace(token_id=seq[len(prefix)-1],hidden=tuple(prefix))
        def draft(prefix,state,position):
            if mode=='reject' or (mode=='reject_third' and position==3):return 800
            n=len(prefix)-1
            return seq[n] if n<len(seq) else 801
        def prepare(row,boundary):
            v=row['prefix_before_ids']+row['candidate_ids'][:row['accepted_count']]
            assert tuple(v)==boundary.hidden
            assert row['prefix_after_ids']==v+[boundary.token_id]
            prepared.append(row['round']);return {'from_round':row['round']}
        result=decode(prompt,draft_token=draft,target_step=target,prepare=prepare,
                      on_round=lambda r,s:events.append(r),oracle_ids=seq,max_new=len(seq),eos=eos)
        assert result['output_ids']==seq and sum(len(r['committed_ids']) for r in events)==len(seq)
        assert result['actual_target_forwards']==len(seq)
        assert all(r['feedback_from_round']==r['round']-1 for r in events[1:])
        assert len(prepared)==len(events)-1
        if mode=='reject':assert result['accepted_draft_tokens']==0
        cases.append({'name':name,'status':'pass','rounds':len(events),'accepted':result['accepted_draft_tokens']})
    return {'status':'pass','GPU_used':False,'cases':cases}

if __name__=='__main__':print(json.dumps(cpu_tests()))
