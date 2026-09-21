import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,time,numpy as np,mujoco,abc_sim
from abc_act.policies.act import select_episodes
from abc_minimal.episode_io import load_episode,load_scene_xml,load_scene_qpos
root=Path('/home/sra/ksagar/abc-act');out=root/'outputs/act-live-training-data';out.mkdir(exist_ok=False)
rng=np.random.default_rng(20260921);manifest={'camera_backend':'mujoco','camera_keys':['top','left','right'],'chunk_size':20,'joint_noise_std':0.01,'gripper_noise_std':0.02,'noise_probability':0.5,'seed':20260921,'splits':{}}
reserved={r['episode'] for r in json.loads((root/'outputs/act_ground_truth_replay.json').read_text())}
for split,pool,episodes,frames in [('train','train_sim',64,128),('val','val_sim',16,32)]:
 paths=select_episodes(root/'abc/cache'/pool,'sim_put_the_plastic_bottles_in_the_bin')
 if split=='val':paths=[p for p in paths if p.name not in reserved]
 paths=[paths[i] for i in np.linspace(0,len(paths)-1,episodes,dtype=int)]
 dest=out/split;dest.mkdir();n=episodes*frames
 images=np.lib.format.open_memmap(dest/'images.npy',mode='w+',dtype=np.uint8,shape=(n,3,3,168,224))
 states_out=np.zeros((n,14),np.float32);actions_out=np.zeros((n,20,14),np.float32);padding=np.ones((n,20),bool);source_t=[];perturbed=[];cursor=0;started=time.monotonic()
 for ep_index,path in enumerate(paths):
  m,states,actions=load_episode(path);qpos=load_scene_qpos(path,m,len(states));xml,_=load_scene_xml(path,m,root/'abc/abc_sim/models/assets')
  env=abc_sim.make_env(task='put_plastic_bottles_in_bin',scene_xml_string=xml,enable_task_randomizer=False,render_cameras=True,camera_backend='mujoco',camera_height=168,camera_width=224,physics_dt=1/(30*17),control_decimation=17)
  try:
   env.reset(randomize=False)
   times=np.linspace(0,len(states)-1,frames,dtype=int)
   for t in times:
    env.data.qpos[:]=qpos[t];env.data.qvel[:]=0;env.data.time=t/30
    noisy=split=='train' and rng.random()<.5
    if noisy:
     state=states[t]+rng.normal(0,.01,14);state[[6,13]]=np.clip(states[t,[6,13]]+rng.normal(0,.02,2),0,1)
     env._set_qpos_from_state(state)
    mujoco.mj_forward(env.model,env.data);obs=env.get_obs()
    images[cursor]=np.stack([obs['images'][cam] for cam in manifest['camera_keys']]);states_out[cursor]=obs['state']
    valid=min(20,len(states)-t);actions_out[cursor,:valid]=actions[t:t+valid];padding[cursor,:valid]=False
    source_t.append(int(t));perturbed.append(noisy);cursor+=1
  finally:env.close()
  print(split,ep_index+1,'/',episodes,'frames',cursor,'seconds',round(time.monotonic()-started,1),flush=True)
 images.flush();np.save(dest/'states.npy',states_out);np.save(dest/'actions.npy',actions_out);np.save(dest/'is_pad.npy',padding)
 manifest['splits'][split]={'episodes':[str(p) for p in paths],'frames_per_episode':frames,'samples':n,'source_timesteps':source_t,'perturbed':perturbed}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',out,flush=True)
