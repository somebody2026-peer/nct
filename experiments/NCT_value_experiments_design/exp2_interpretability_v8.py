# -*- coding: utf-8 -*-
"""
Experiment 2: Interpretability Validation (V8 - Deep Debug)
=============================================================
Exp-2: Interpretability Validation - Debug Attention Maps Structure

V8 Goal:
- Understand the actual structure of attention_maps
- Find correct way to extract per-head activations
- Achieve differentiated head activations

V7 Result:
- All heads = 0.500 (uniform)
- Problem: aggregation method may be wrong

Version History:
- V6: Threshold fixed, 100% winner rate
- V7: Clean logs, but uniform activations
- V8: Debug attention structure
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


class Exp2V8:
    """Interpretability Validation Experiment V8 - Deep Debug"""
    
    def __init__(self, threshold=0.3):
        self.use_gpu = False
        self.device = 'cpu'
        print(f"Using device: {self.device}")
        print(f"Target Consciousness Threshold: {threshold}")
        
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
        if hasattr(self.nct, 'attention_workspace'):
            self.nct.attention_workspace.consciousness_threshold = threshold
            print(f"✅ Threshold set to: {threshold}")
        
        self.nct.to(self.device)
        self.nct.eval()
        
        self.version = f"v8-DeepDebug{threshold}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path('results') / f'exp2_interpretability_{self.version}_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.extraction_stats = {
            'real_attention_maps': 0,
            'workspace_info': 0,
            'fallback': 0,
            'total_trials': 0
        }
    
    def debug_attention_structure(self, attn_maps, trial_num=0):
        """V8: Debug the structure of attention_maps
        
        V8 Fix: Use numpy methods instead of torch
        """
        print(f"\n{'='*60}")
        print(f"V8 DEBUG: Attention Maps Structure (Trial {trial_num})")
        print(f"{'='*60}")
        
        # Check type
        print(f"Type: {type(attn_maps)}")
        
        # Check if array/tensor
        if hasattr(attn_maps, 'shape'):
            shape = attn_maps.shape
            print(f"Shape: {shape}")
            print(f"Dtype: {attn_maps.dtype}")
            
            # V8 FIX: Use numpy methods
            arr = np.array(attn_maps)
            print(f"Min: {arr.min():.6f}")
            print(f"Max: {arr.max():.6f}")
            print(f"Mean: {arr.mean():.6f}")
            print(f"Std: {arr.std():.6f}")
            
            # Check if shape is [B, H, L, L]
            if len(shape) == 4:
                B, H, L1, L2 = shape
                print(f"\nInterpretation: [Batch={B}, Heads={H}, Query={L1}, Key={L2}]")
                
                # Check per-head statistics
                print(f"\nPer-Head Statistics:")
                for h in range(H):
                    head_attn = arr[0, h, :, :]  # First batch item
                    h_min = head_attn.min()
                    h_max = head_attn.max()
                    h_mean = head_attn.mean()
                    h_std = head_attn.std()
                    h_sum = head_attn.sum()
                    print(f"  Head {h}: min={h_min:.4f}, max={h_max:.4f}, mean={h_mean:.4f}, std={h_std:.4f}, sum={h_sum:.2f}")
                
                # V8 KEY: Use STD as activation metric
                print(f"\n{'='*60}")
                print("V8: STD-Based Activation (KEY METRIC)")
                print(f"{'='*60}")
                
                head_stds = []
                for h in range(H):
                    head_attn = arr[0, h, :, :]
                    std_val = head_attn.std()
                    head_stds.append(std_val)
                
                print(f"Head STDs (activation metric): {head_stds}")
                print(f"\n✅ V8: Using STD as activation metric!")
                print(f"   Higher STD = More selective attention = More active head")
                
                return np.array(head_stds)
        
        return np.ones(8) * 0.5
    
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
    
    def extract_head_activations(self, state, trial_num=0, debug=False):
        """Extract 8-head activation pattern with optional debug
        
        V8 Fix: attention_maps is already numpy array, no .detach() needed
        V8 Discovery: Use STD as activation metric (higher = more selective attention)
        """
        self.extraction_stats['total_trials'] += 1
        
        try:
            if hasattr(state, 'workspace_content') and state.workspace_content is not None:
                wc = state.workspace_content
                
                if hasattr(wc, 'attention_maps') and wc.attention_maps is not None:
                    attn_maps = wc.attention_maps
                    
                    # V8 FIX: Handle both torch tensor and numpy array
                    if hasattr(attn_maps, 'detach'):
                        attn_maps = attn_maps.detach().cpu().numpy()
                    elif isinstance(attn_maps, np.ndarray):
                        pass  # Already numpy array
                    else:
                        attn_maps = np.array(attn_maps)
                    
                    # V8: Debug first trial
                    if debug:
                        head_act = self.debug_attention_structure(attn_maps, trial_num)
                        self.extraction_stats['real_attention_maps'] += 1
                        return head_act
                    
                    # V8 KEY FIX: Use STD as activation metric
                    # Higher std = more selective attention = more active head
                    if len(attn_maps.shape) == 4:
                        # Calculate std per head (over query and key dimensions)
                        head_stds = []
                        for h in range(attn_maps.shape[1]):
                            head_attn = attn_maps[0, h, :, :]  # [Query, Key]
                            # Use std as activation metric
                            std_val = np.std(head_attn)
                            head_stds.append(std_val)
                        
                        head_activations = np.array(head_stds)
                        if len(head_activations) == 8:
                            self.extraction_stats['real_attention_maps'] += 1
                            return head_activations
            
            self.extraction_stats['fallback'] += 1
            return np.ones(8) * 0.5
            
        except Exception as e:
            print(f"  [Error] Extraction failed: {e}")
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
                
                # V8: Debug first trial of first case only
                debug = (i == 0 and case_name == "Visual Salience Detection")
                head_act = self.extract_head_activations(state, i, debug=debug)
                
                top2 = np.argsort(head_act)[-2:][::-1]
                match = len(set(top2) & set(expected_heads)) / 2.0
                
                trial_acts.append(head_act)
                trial_matches.append(match)
                
            except Exception as e:
                print(f"  [Error] Trial {i}: {e}")
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
        print(f"  Head Activations:")
        for h in range(8):
            bar = '█' * int(mean_act[h] * 20)
            marker = ' ← EXPECTED' if h in expected_heads else ''
            print(f"    Head {h}: {bar} ({mean_act[h]:.3f} ± {std_act[h]:.3f}){marker}")
        
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
        print("Experiment 2: Interpretability Validation (V8 - Deep Debug)")
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
        
        print(f"\n{'='*80}")
        print("OVERALL RESULTS (V8)")
        print(f"{'='*80}")
        print(f"Total Match Score: {total_match:.2%}")
        print(f"Avg Winner State Rate: {avg_winner_rate:.1%}")
        
        report = {
            'experiment': 'Exp-2: Interpretability Validation',
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'overall_statistics': {
                'total_match_score': float(total_match),
                'avg_winner_rate': float(avg_winner_rate)
            },
            'extraction_statistics': self.extraction_stats
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
            ax.set_ylabel('Activation Strength', fontsize=11)
            ax.set_title(f"{key.replace('_', ' ').title()}\nMatch: {result['match_score']:.0%}", fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'H{i}' for i in range(8)])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('NCT Interpretability Validation (V8)\nDeep Debug', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        viz_path = self.results_dir / 'interpretability_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")


def main():
    print("="*80)
    print("Experiment 2 V8: NCT Interpretability Validation (Deep Debug)")
    print("V8: Debug attention_maps structure to find correct extraction method")
    print("="*80)
    
    exp = Exp2V8(threshold=0.3)
    report = exp.run()
    
    print("\n" + "="*80)
    print("V8 COMPLETE - Check debug output above for attention structure")
    print("="*80)


if __name__ == '__main__':
    main()
