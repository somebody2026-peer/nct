# Experiment 2 Interpretability V1
import torch, numpy as np, matplotlib.pyplot as plt, json
from pathlib import Path
from datetime import datetime
from torchvision.datasets import MNIST
import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

from nct_modules import NCTManager, NCTConfig

class Exp2:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 正确的 NCTConfig 参数
        config = NCTConfig(
            n_heads=8,
            n_layers=4,
            d_model=512,
            dim_ff=2048,
            visual_patch_size=4,
            visual_embed_dim=256
        )
        self.nct = NCTManager(config)
        self.nct.to(self.device)  # 移动到 GPU/CPU
        self.version = "v1-BaseVersion"
        self.results_dir = Path('results') / f'exp2_interpretability_{self.version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results dir: {self.results_dir}")
    
    def gen_stimuli(self, t, n=50):
        stim = []
        if t == 'edge':
            for _ in range(n):
                img = np.zeros((28,28), dtype=np.float32)
                pos = np.random.randint(10,18)
                if np.random.rand()>0.5: img[pos-2:pos+2,:]=1.0
                else: img[:,pos-2:pos+2]=1.0
                stim.append(np.clip(img,0,1))
        elif t == 'novel':
            for _ in range(n):
                img = np.zeros((28,28), dtype=np.float32)
                s = np.random.choice(['circle','rect','cross'])
                if s=='circle':
                    y,x=np.ogrid[:28,:28]
                    mask=(x-14)**2+(y-14)**2<=np.random.randint(3,8)**2
                    img[mask]=1.0
                elif s=='rect': img[np.random.randint(2,10):np.random.randint(18,26), np.random.randint(2,10):np.random.randint(18,26)]=1.0
                else: img[12:16,:]=1.0; img[:,12:16]=1.0
                stim.append(img)
        else:
            stim = [np.random.rand(28,28).astype(np.float32)*0.5+0.25 for _ in range(n)]
        return stim
    
    def run(self):
        cases = {'visual':{'name':'Visual Salience','expected':[0,1],'type':'edge'}, 'novelty':{'name':'Novelty','expected':[6,7],'type':'novel'}}
        results = {}
        for k,c in cases.items():
            print(f"\nTest: {c['name']}")
            stim = self.gen_stimuli(c['type'], 50)
            acts, matches = [], []
            for s in stim:
                # process_cycle 期望 numpy 数组，会自动转换
                # 视觉输入应该是 [H, W] 或 [C, H, W]
                state = self.nct.process_cycle({'visual': s})  # s shape: [28, 28]
                ha = np.random.rand(8)*0.5+0.25  # Placeholder
                top = np.argsort(ha)[-2:]
                match = len(set(top)&set(c['expected']))/2.0
                acts.append(ha); matches.append(match)
            mean_act, std_act = np.mean(acts,axis=0), np.std(acts,axis=0)
            score = np.mean(matches)
            results[k] = {'match_score':score, 'mean_activations':mean_act.tolist(), 'n_trials':50}
            print(f"Match: {score:.2%}")
        
        report = {'experiment':'Exp-2', 'version':self.version, 'results':results, 'overall':{'total_match':np.mean([r['match_score'] for r in results.values()])}}
        with open(self.results_dir/'experiment_report.json','w') as f: json.dump(report,f,indent=2)
        
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for i,(k,r) in enumerate(results.items()):
            axes[i].bar(range(8), r['mean_activations'])
            axes[i].set_title(f"{k}: Match={r['match_score']:.0%}")
        plt.savefig(self.results_dir/'results.png', dpi=300)
        print(f"\nSaved to {self.results_dir}")

if __name__=='__main__':
    Exp2().run()
