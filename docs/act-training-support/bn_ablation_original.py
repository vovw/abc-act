from pathlib import Path
import torch,json,numpy as np
from torch.utils.data import DataLoader,Subset
from abc_act.policies.act import ACTPolicy,ACTDataset,evaluate
root=Path('/home/sra/ksagar/abc-act');torch.set_num_threads(4)
c=torch.load(root/'checkpoints/bottles-sim-augmentation/last.pt',map_location='cpu',weights_only=True)
base=torch.load(root/'checkpoints/bottles-sim-recovery/eval-step-5000.pt',map_location='cpu',weights_only=True)
out=root/'outputs/act-bn-ablation';out.mkdir(exist_ok=True)
torch.save(c,out/'augmented_stats.pt')
d=ACTDataset(c['val_episodes'],c['model_config']['chunk_size'],c['norm_stats'])
loader=DataLoader(Subset(d,np.linspace(0,len(d)-1,512,dtype=int).tolist()),batch_size=8,num_workers=0)
p=ACTPolicy(**c['model_config']).cuda();p.load_state_dict(c['model']);a=evaluate(p,loader,torch.device('cuda'));print('augmented stats val',a,flush=True)
keys=[k for k in c['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
for k in keys:c['model'][k]=base['model'][k]
p.load_state_dict(c['model']);b=evaluate(p,loader,torch.device('cuda'));print('original stats val',b,flush=True)
c['ablation']={'description':'Augmented weights with pre-augmentation batchnorm running statistics','source_step':c['step'],'stats_source':'checkpoints/bottles-sim-recovery/eval-step-5000.pt'}
torch.save(c,out/'original_stats.pt')
(out/'results.json').write_text(json.dumps(dict(source_step=c['step'],augmented_stats_val=a,original_stats_val=b,changed_buffers=len(keys)),indent=2))
