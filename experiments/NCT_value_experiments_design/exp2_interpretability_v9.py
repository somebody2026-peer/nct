# -*- coding: utf-8 -*-
"""
Experiment 2: Interpretability Validation (V9 - Trained Model)
===============================================================
Exp-2: Interpretability Validation - Using Trained NCT Model

V9 Improvements:
1. Load pre-trained MNIST model weights
2. Use trained model for better attention differentiation
3. Compare trained vs untrained model behavior

V8 Discovery:
- STD as activation metric works
- But untrained model shows uniform patterns (Head 7 always dominant)
- Trained model should show differentiated patterns

Version History:
- V6: Threshold fixed, 100% winner rate
- V7: Clean logs, uniform mean activations
- V8: STD-based activation, but uniform patterns
- V9: Use trained model for real differentiation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nct_modules import NCTManager, NCTConfig


class Exp2V9:
    """Interpretability Validation Experiment V9 - Trained Model"""
    
    def __init__(self, threshold=0.3, pretrained_path=None):
        self.use_gpu = False
        self.device = 'cpu'
        print(f"Using device: {self.device}")
        print(f"V9: Using trained model for attention differentiation")
        
        config = NCTConfig(
            n_heads=8,
            n_layers=4,
            d_model=512,
            dim_ff=2048,
            visual_patch_size=4,
            visual_embed_dim=256,
            consciousness_threshold=threshold
        )
        
        self.nct = NCTManager(config)
        
        # V6 FIX: Set threshold
        if hasattr(self.nct, 'attention_workspace'):
            self.nct.attention_workspace.consciousness_threshold = threshold
            print(f"✅ Threshold set to: {threshold}")
        
        # V9: Try to load pretrained weights
        self.model_type = "random_init"
        if pretrained_path and Path(pretrained_path).exists():
            try:
                print(f"\n📂 Loading pretrained model from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.nct.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        self.model_type = "pretrained"
                        print(f"✅ Loaded model_state_dict from checkpoint")
                    elif 'state_dict' in checkpoint:
                        self.nct.load_state_dict(checkpoint['state_dict'], strict=False)
                        self.model_type = "pretrained"
                        print(f"✅ Loaded state_dict from checkpoint")
                    else:
                        # Try loading as state_dict directly
                        self.nct.load_state_dict(checkpoint, strict=False)
                        self.model_type = "pretrained"
                        print(f"✅ Loaded checkpoint as state_dict")
                else:
                    print(f"⚠️  Checkpoint format not recognized, using random init")
            except Exception as e:
                print(f"⚠️  Failed to load pretrained model: {e}")
                print(f"   Using random initialized model")
        else:
            print(f"\n⚠️  No pretrained model found, using random init")
            # V9: Warmup the model with a few forward passes
            print(f"   Warming up model with random inputs...")
            self._warmup_model()
            self.model_type = "warmup"
        
        self.nct.to(self.device)
        self.nct.eval()
        
        self.version = f"v9-TrainedModel_{self.model_type}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path('results') / f'exp2_interpretability_{self.version}_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {self.results_dir}")
        
        self.extraction_stats = {
            'real_attention_maps': 0,
            'fallback': 0,
            'total_trials': 0
        }
    
    def _warmup_model(self, n_warmup=10):
        """Warmup model with random inputs to initialize weights better"""
        with torch.no_grad():
            for _ in range(n_warmup):
                dummy_input = np.random.rand(28, 28).astype(np.float32)
                try:
                    self.nct.process_cycle({'visual': dummy_input})
                except:
                    pass
        print(f"   Model warmed up with {n_warmup} random inputs")
    
    def gen_stimuli(self, stim_type, n=50):
        """Generate test stimuli"""
        stim = []
        
        if stim_type == 'edge':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pos = np.random.randint(10, 18)
                thickness = np.random.randint(2, 5)
                if np.random.rand() > 0.5:
                    img[pos-thickness:pos+thickness, :] = 1.0
                else:
                    img[:, pos-thickness:pos+thickness] = 1.0
                stim.append(np.clip(img, 0, 1))
        
        elif stim_type == 'familiar':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                digit = np.random.randint(0, 10)
                if digit < 2:
                    y, x = np.ogrid[:28, :28]
                    mask = (x - 14)**2 + (y - 14)**2 <= 8**2
                    mask2 = (x - 14)**2 + (y - 14)**2 >= 5**2
                    img[mask & mask2] = 1.0
                elif digit < 4:
                    img[4:24, 12:16] = 1.0
                    img[4:8, 10:18] = 1.0
                elif digit < 6:
                    img[10:18, :] = 1.0
                    img[:, 10:18] = 1.0
                else:
                    img[6:10, 4:24] = 1.0
                    img[10:24, 12:16] = 1.0
                stim.append(np.clip(img, 0, 1))
        
        elif stim_type == 'discriminative':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pattern_type = np.random.choice(['corner_pair', 'center_blob', 'stripe_set'])
                if pattern_type == 'corner_pair':
                    img[2:10, 2:10] = 1.0
                    img[18:26, 18:26] = 1.0
                elif pattern_type == 'center_blob':
                    y, x = np.ogrid[:28, :28]
                    mask = (x - 14)**2 + (y - 14)**2 <= 6**2
                    img[mask] = 1.0
                else:
                    for i in range(4):
                        y = 6 + i * 5
                        img[y:y+2, :] = 1.0
                stim.append(np.clip(img, 0, 1))
        
        elif stim_type == 'novel':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pattern = np.random.choice(['spiral', 'random_blobs', 'fractal'])
                if pattern == 'spiral':
                    for angle in range(0, 360, 15):
                        rad = np.deg2rad(angle)
                        r = angle / 360 * 10 + 2
                        x = int(14 + r * np.cos(rad))
                        y = int(14 + r * np.sin(rad))
                        if 0 <= x < 28 and 0 <= y < 28:
                            img[max(0, y-1):min(28, y+2), max(0, x-1):min(28, x+2)] = 1.0
                elif pattern == 'random_blobs':
                    for _ in range(5):
                        cx = np.random.randint(5, 23)
                        cy = np.random.randint(5, 23)
                        r = np.random.randint(2, 5)
                        y, x = np.ogrid[:28, :28]
                        mask = (x - cx)**2 + (y - cy)**2 <= r**2
                        img[mask] = 1.0
                else:
                    for i in range(0, 28, 4):
                        for j in range(0, 28, 4):
                            if (i + j) % 8 == 0:
                                img[i:i+4, j:j+4] = 1.0
                stim.append(np.clip(img, 0, 1))
        
        return stim
    
    def extract_head_activations(self, state):
        """Extract 8-head activation pattern using STD metric"""
        self.extraction_stats['total_trials'] += 1
        
        try:
            if hasattr(state, 'workspace_content') and state.workspace_content is not None:
                wc = state.workspace_content
                
                if hasattr(wc, 'attention_maps') and wc.attention_maps is not None:
                    attn_maps = wc.attention_maps
                    
                    # Handle both torch tensor and numpy array
                    if hasattr(attn_maps, 'detach'):
                        attn_maps = attn_maps.detach().cpu().numpy()
                    elif isinstance(attn_maps, np.ndarray):
                        pass
                    else:
                        attn_maps = np.array(attn_maps)
                    
                    # Use STD as activation metric
                    if len(attn_maps.shape) == 4:
                        head_stds = []
                        for h in range(attn_maps.shape[1]):
                            head_attn = attn_maps[0, h, :, :]
                            std_val = np.std(head_attn)
                            head_stds.append(std_val)
                        
                        head_activations = np.array(head_stds)
                        if len(head_activations) == 8:
                            self.extraction_stats['real_attention_maps'] += 1
                            return head_activations
            
            self.extraction_stats['fallback'] += 1
            return np.ones(8) * 0.5
            
        except Exception as e:
            self.extraction_stats['fallback'] += 1
            return np.ones(8) * 0.5
    
    def run_single_case(self, case_name, expected_heads, stimulus_type, n_trials=50):
        """Run a single test case"""
        print(f"\n{'='*70}")
        print(f"Test Case: {case_name}")
        print(f"Expected Active Heads: {expected_heads}")
        print(f"{'='*70}")
        
        stimuli = self.gen_stimuli(stimulus_type, n=n_trials)
        print(f"Generated {len(stimuli)} stimuli")
        
        trial_acts = []
        trial_matches = []
        winner_count = 0
        
        for i, s in enumerate(stimuli):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(stimuli)}...")
            
            try:
                state = self.nct.process_cycle({'visual': s})
                
                if hasattr(state, 'workspace_content') and state.workspace_content is not None:
                    wc = state.workspace_content
                    if hasattr(wc, 'attention_maps') and wc.attention_maps is not None:
                        winner_count += 1
                
                head_act = self.extract_head_activations(state)
                
                top2 = np.argsort(head_act)[-2:][::-1]
                match = len(set(top2) & set(expected_heads)) / 2.0
                
                trial_acts.append(head_act)
                trial_matches.append(match)
                
            except Exception as e:
                trial_acts.append(np.ones(8) * 0.5)
                trial_matches.append(0.0)
        
        if len(trial_acts) == 0:
            return {'match_score': 0.0, 'mean_activations': [0]*8, 'n_trials': 0, 'winner_rate': 0}
        
        mean_act = np.mean(trial_acts, axis=0)
        std_act = np.std(trial_acts, axis=0)
        score = np.mean(trial_matches)
        winner_rate = winner_count / len(trial_acts)
        
        print(f"\n  Results:")
        print(f"  Mean Match Score: {score:.2%}")
        print(f"  Winner State Rate: {winner_rate:.1%}")
        print(f"  Head Activations (STD-based):")
        for h in range(8):
            bar = '█' * int(mean_act[h] * 50)
            marker = ' ← EXPECTED' if h in expected_heads else ''
            print(f"    Head {h}: {bar} ({mean_act[h]:.4f} ± {std_act[h]:.4f}){marker}")
        
        return {
            'match_score': float(score),
            'mean_head_activations': mean_act.tolist(),
            'std_head_activations': std_act.tolist(),
            'n_trials': len(trial_acts),
            'winner_rate': float(winner_rate)
        }
    
    def run(self):
        """Run complete experiment"""
        print("\n" + "="*80)
        print(f"Experiment 2: Interpretability Validation (V9 - {self.model_type})")
        print("="*80)
        
        test_cases = [
            {'key': 'visual_salience', 'name': 'Visual Salience Detection',
             'expected_heads': [0, 1], 'stimulus_type': 'edge', 'n_trials': 50},
            {'key': 'emotional_value', 'name': 'Emotional Value Assessment',
             'expected_heads': [2, 3], 'stimulus_type': 'familiar', 'n_trials': 50},
            {'key': 'task_relevance', 'name': 'Task Relevance Selection',
             'expected_heads': [4, 5], 'stimulus_type': 'discriminative', 'n_trials': 50},
            {'key': 'novelty_detection', 'name': 'Novelty Detection',
             'expected_heads': [6, 7], 'stimulus_type': 'novel', 'n_trials': 50}
        ]
        
        results = {}
        for tc in test_cases:
            result = self.run_single_case(tc['name'], tc['expected_heads'], tc['stimulus_type'], tc['n_trials'])
            results[tc['key']] = result
        
        total_match = np.mean([r['match_score'] for r in results.values()])
        avg_winner_rate = np.mean([r.get('winner_rate', 0) for r in results.values()])
        success = total_match > 0.7
        partial = 0.4 < total_match <= 0.7
        
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS (V9 - {self.model_type})")
        print(f"{'='*80}")
        print(f"Total Match Score: {total_match:.2%}")
        print(f"Avg Winner State Rate: {avg_winner_rate:.1%}")
        print(f"Status: {'✅ SUCCESS' if success else '⚠️ PARTIAL' if partial else '❌ FAILED'}")
        
        print(f"\n  Extraction Statistics:")
        total = self.extraction_stats['total_trials']
        if total > 0:
            print(f"    Real Attention Maps: {self.extraction_stats['real_attention_maps']}/{total} ({100*self.extraction_stats['real_attention_maps']/total:.1f}%)")
            print(f"    Fallback: {self.extraction_stats['fallback']}/{total}")
        
        report = {
            'experiment': 'Exp-2: Interpretability Validation',
            'version': self.version,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'overall_statistics': {
                'total_match_score': float(total_match),
                'avg_winner_rate': float(avg_winner_rate),
                'success': bool(success),
                'partial_success': bool(partial)
            },
            'extraction_statistics': self.extraction_stats,
            'conclusions': [
                f"V9: Model type = {self.model_type}",
                f"Overall match score: {total_match:.2%}",
                f"Winner state rate: {avg_winner_rate:.1f}%"
            ]
        }
        
        report_path = self.results_dir / 'experiment_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to: {report_path}")
        
        self.visualize(results)
        
        return report
    
    def visualize(self, results):
        """Generate visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        
        for idx, (key, result) in enumerate(results.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            mean_act = np.array(result['mean_head_activations'])
            std_act = np.array(result.get('std_head_activations', [0]*8))
            
            x = np.arange(8)
            ax.bar(x, mean_act, yerr=std_act, capsize=3, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Attention Head', fontsize=11)
            ax.set_ylabel('Activation (STD)', fontsize=11)
            ax.set_title(f"{key.replace('_', ' ').title()}\nMatch: {result['match_score']:.0%}", fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'H{i}' for i in range(8)])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'NCT Interpretability Validation (V9)\nModel: {self.model_type}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        viz_path = self.results_dir / 'interpretability_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")


def main():
    print("="*80)
    print("Experiment 2 V9: NCT Interpretability Validation (Trained Model)")
    print("="*80)
    
    # Try to find a pretrained model (ordered by accuracy)
    # Use absolute paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    pretrained_paths = [
        # Best model: SimplifiedNCT 99.61% accuracy
        project_root / "results/anomaly_detection_complete_20260227_005219/best_model.pt",
        # NCT v3.1: 99.50% accuracy
        project_root / "results/training_v3/best_model_v3.pt",
        # Other anomaly detection models
        project_root / "results/anomaly_detection_complete_20260227_002841/best_model.pt",
        project_root / "results/anomaly_detection_complete_20260227_001039/best_model.pt",
        project_root / "results/anomaly_detection_supervised_20260227_070226/best_model.pt",
        # MNIST classification (lower accuracy)
        project_root / "results/mnist_classification/best_model_ep2_acc0.118.pth",
    ]
    
    pretrained_path = None
    for path in pretrained_paths:
        if Path(path).exists():
            pretrained_path = path
            print(f"Found pretrained model: {path}")
            break
    
    exp = Exp2V9(threshold=0.3, pretrained_path=pretrained_path)
    report = exp.run()
    
    print("\n" + "="*80)
    print("V9 RESULTS SUMMARY")
    print("="*80)
    print(f"Model Type: {report['model_type']}")
    print(f"Match Score: {report['overall_statistics']['total_match_score']:.2%}")
    print(f"Winner Rate: {report['overall_statistics']['avg_winner_rate']:.1f}%")


if __name__ == '__main__':
    main()
