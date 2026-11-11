from pathlib import Path
from datetime import datetime, timezone
import csv,json,os,subprocess,time,traceback

ROOT=Path('/nfs/home/suv6790/mlds490_hw2')
OUT=ROOT/'results/upgrade_comparison'
RUN=ROOT/'results/mspacman_extended'
PY=ROOT/'.venv312/bin/python'
PID=4050763
DEADLINE=time.monotonic()+6*3600
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
COMMON=['--baseline-run','results/mspacman','--new-run','results/mspacman_extended','--output','results/upgrade_comparison']

def status(phase,**extra):
    data={'phase':phase,'watcher_pid':os.getpid(),'main_pid':PID,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra}
    temporary=OUT/'completion_status.tmp'
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(OUT/'completion_status.json')
    print(json.dumps(data),flush=True)

def remaining():
    seconds=DEADLINE-time.monotonic()
    if seconds<=0:
        raise TimeoutError('Completion watcher six-hour bound expired; main training process is untouched')
    return seconds

def wait_for(predicate,label):
    status(label)
    while not predicate():
        remaining()
        time.sleep(min(10,remaining()))

def development_saved():
    if not Path('/proc',str(PID)).exists():
        raise RuntimeError('Main process ended before development-only selection was frozen')
    return 'Development at episode 6000:' in (RUN/'run.log').read_text()

def final_saved():
    if not (RUN/'metrics.json').exists() or not (RUN/'evaluation.csv').exists():
        if not Path('/proc',str(PID)).exists():
            raise RuntimeError('Main process ended without final evaluation outputs')
        return False
    try:
        metrics=json.loads((RUN/'metrics.json').read_text())
        with (RUN/'evaluation.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        return len(rows)==500 and {int(row['seed']) for row in rows}==set(range(200007,200507)) and metrics['evaluation']['n']==500 and not Path('/proc',str(PID)).exists()
    except (ValueError,KeyError):
        return False

def execute(phase,args,log):
    status(phase,log=str(log.relative_to(ROOT)))
    with log.open('w') as handle:
        result=subprocess.run([str(PY),*args],cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,timeout=min(3600,remaining()),check=False)
    if result.returncode:
        raise RuntimeError(phase+' exited '+str(result.returncode)+'; see '+str(log))
    status(phase+'_complete',exit_code=result.returncode)

def verify_json(path):
    if not path.is_file() or path.stat().st_size==0:
        raise RuntimeError('Missing output '+str(path))
    return json.loads(path.read_text())

def verify_csv(path,expected=None):
    with path.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows or (expected is not None and len(rows)!=expected):
        raise RuntimeError('Wrong row count '+str(path))

def main():
    descriptor=os.open(OUT/'completion_watcher.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,'w') as handle:
        handle.write(str(os.getpid()))
    if (OUT/'selection_decision.json').exists() or (RUN/'COMPLETE').exists():
        raise RuntimeError('Existing completed/frozen outputs require review; refusing duplicate processing')
    wait_for(development_saved,'waiting_development_6000')
    execute('freeze_selection',['compare_runs.py',*COMMON,'--freeze-only'],OUT/'selection_decision.log')
    frozen=verify_json(OUT/'selection_decision.json')
    status('selection_frozen',selection=frozen['selection'],frozen_at_utc=frozen['frozen_at_utc'],candidate_checkpoint_sha256=frozen['evidence_sha256']['candidate_checkpoint'])
    wait_for(final_saved,'waiting_final_500')
    execute('paired_comparison',['compare_runs.py',*COMMON,'--device','cuda'],OUT/'comparison.log')
    result=verify_json(OUT/'comparison.json')
    if result['weights_unchanged'] is not True:
        raise RuntimeError('Comparison weights changed')
    verify_csv(OUT/'baseline_evaluation.csv',500)
    verify_csv(OUT/'random_evaluation.csv',500)
    execute('validation',['validate.py','--task','mspacman','--run','results/mspacman_extended','--episodes','50','--device','cuda'],RUN/'validation.log')
    verify_json(RUN/'validation.json')
    for name in ('final','initial','random','blackout'):
        verify_csv(RUN/('validation_'+name+'.csv'),50)
    verify_json(RUN/'validation_first10_steps.json')
    execute('scientific_validation',['scientific_validation.py','--run','results/mspacman_extended','--episodes','50','--device','cuda'],RUN/'scientific_validation.log')
    scientific=verify_json(RUN/'scientific_validation.json')
    if scientific['weights_unchanged'] is not True:
        raise RuntimeError('Scientific validation weights changed')
    for name in ('calibration','ending_targets','calibration_states','scientific_repeat_latest','scientific_constant128'):
        verify_csv(RUN/(name+'.csv'),50 if name in ('calibration','scientific_repeat_latest','scientific_constant128') else None)
    (RUN/'COMPLETE').touch(exist_ok=False)
    status('complete',complete_file=str((RUN/'COMPLETE').relative_to(ROOT)))

try:
    main()
except BaseException as error:
    status('failed',error=str(error))
    traceback.print_exc()
    raise
