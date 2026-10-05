from dataclasses import replace
import json
from pathlib import Path
import torch
from controllers import load_bundle
from evaluate import run_evaluation

torch.set_num_threads(1)
policy,_,config,_=load_bundle('runs/imitation_complete/best.pt','cpu')
result=run_evaluation(policy,None,replace(config,physics_dt=.0025,action_repeat=8),'cpu',64,seed=83000,mode='policy')
Path('logs/imitation_final_half_step.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['summary'],indent=2))
