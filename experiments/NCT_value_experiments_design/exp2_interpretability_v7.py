# -*- coding: utf-8 -*-
"""
Experiment 2: Interpretability Validation (V7 - Clean & Optimized)
===================================================================
Exp-2: Interpretability Validation - Clean Version with Optimized Stimuli

V7 Improvements:
1. Fix device mismatch errors by using CPU (cleaner logs)
2. Optimized stimuli for stronger differentiation between head activations
3. Enhanced stimulus contrast and salience

V6 Breakthrough:
- Fixed threshold bug: 100% winner rate, 100% real attention extraction
- But all head activations were uniform (0.500)

V7 Goal:
- Differentiate head activations based on stimulus type
- Achieve > 40% match score

Version History:
- V1-V5: Threshold not effective, 0% winner rate
- V6: Threshold fixed, 100% winner rate, uniform activations
- V7: Optimized stimuli, differentiated activations
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


class Exp2V7:
    """Interpretability Validation Experiment V7 - Clean & Optimized"""
    
    def __init__(self, threshold=0.3):
        # V7: Use CPU to avoid device mismatch errors
        self.use_gpu = False
        self.device = 'cpu'
        print(f"Using device: {self.device} (V7: CPU mode for cleaner logs)")
        print(f"Target Consciousness Threshold: {threshold}")
        
        # NCTConfig
        config = NCTConfig(
            n_heads=8,
            n_layers=4,
            d_model=512,
            dim_ff=2048,
            visual_patch_size=4,
            visual_embed_dim=256,
            consciousness_threshold=threshold
        )
        
        # Create NCTManager
        self.nct = NCTManager(config)
        
        # V6 FIX: Directly modify attention_workspace threshold
        if hasattr(self.nct, 'attention_workspace'):
            self.nct.attention_workspace.consciousness_threshold = threshold
            print(f"✅ Threshold set to: {threshold}")
        
        self.nct.to(self.device)
        self.nct.eval()
        
        self.version = f"v7-Optimized{threshold}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path('results') / f'exp2_interpretability_{self.version}_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {self.results_dir}")
        
        # Track statistics
        self.extraction_stats = {
            'real_attention_maps': 0,
            'workspace_info': 0,
            'fallback': 0,
            'total_trials': 0
        }
    
    def gen_stimuli(self, stim_type, n=50):
        """Generate test stimuli - V7 Enhanced for stronger differentiation
        
        V7 Strategy:
        - Edge: Strong horizontal/vertical edges (maximize Head 0-1)
        - Familiar: MNIST-like digits with semantic content (maximize Head 2-3)
        - Discriminative: Unique patterns with task-relevant features (maximize Head 4-5)
        - Novel: OOD patterns with high anomaly (maximize Head 6-7)
        """
        stim = []
        
        if stim_type == 'edge':
            # V7: STRONG edges with high contrast
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pos = np.random.randint(10, 18)
                thickness = np.random.randint(2, 5)  # Thicker edges
                
                if np.random.rand() > 0.5:
                    # Horizontal edge with gradient
                    img[pos-thickness:pos+thickness, :] = 1.0
                    # Add gradient for stronger salience
                    for i in range(3):
                        img[pos-thickness-i:pos-thickness-i+1, :] = 0.8 - i*0.2
                        img[pos+thickness+i:pos+thickness+i+1, :] = 0.8 - i*0.2
                else:
                    # Vertical edge with gradient
                    img[:, pos-thickness:pos+thickness] = 1.0
                    for i in range(3):
                        img[:, pos-thickness-i:pos-thickness-i+1] = 0.8 - i*0.2
                        img[:, pos+thickness+i:pos+thickness+i+1] = 0.8 - i*0.2
                
                stim.append(np.clip(img, 0, 1))
        
        elif stim_type == 'familiar':
            # V7: MNIST-like patterns with SEMANTIC content
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                digit = np.random.randint(0, 10)
                
                # Create more realistic digit patterns
                if digit < 2:
                    # "0" - circle
                    y, x = np.ogrid[:28, :28]
                    mask = (x - 14)**2 + (y - 14)**2 <= 8**2
                    mask2 = (x - 14)**2 + (y - 14)**2 >= 5**2
                    img[mask & mask2] = 1.0
                elif digit < 4:
                    # "1" - vertical line
                    img[4:24, 12:16] = 1.0
                    img[4:8, 10:18] = 1.0  # top
                elif digit < 6:
                    # "+" - cross
                    img[10:18, :] = 1.0
                    img[:, 10:18] = 1.0
                elif digit < 8:
                    # "T" shape
                    img[6:10, 4:24] = 1.0
                    img[10:24, 12:16] = 1.0
                else:
                    # "L" shape
                    img[6:24, 6:10] = 1.0
                    img[20:24, 6:22] = 1.0
                
                stim.append(np.clip(img, 0, 1))
        
        elif stim_type == 'discriminative':
            # V7: UNIQUE patterns with TASK-RELEVANT features
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pattern_type = np.random.choice(['corner_pair', 'center_blob', 'stripe_set', 'checkerboard'])
                
                if pattern_type == 'corner_pair':
                    # Two opposite corners
                    img[2:10, 2:10] = 1.0
                    img[18:26, 18:26] = 1.0
                elif pattern_type == 'center_blob':
                    # Central blob with ring
                    y, x = np.ogrid[:28, :28]
                    mask = (x - 14)**2 + (y - 14)**2 <= 6**2
                    img[mask] = 1.0
                elif pattern_type == 'stripe_set':
                    # Multiple horizontal stripes
                    for i in range(4):
                        y = 6 + i * 5
                        img[y:y+2, :] = 1.0
                else:  # checkerboard
                    # Small checkerboard pattern
                    for i in range(0, 28, 4):
                        for j in range(0, 28, 4):
                            if (i + j) % 8 == 0:
                                img[i:i+4, j:j+4] = 1.0
                
                stim.append(np.clip(img, 0, 1))
        
        elif stim_type == 'novel':
            # V7: HIGHLY NOVEL/ANOMALOUS patterns
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pattern = np.random.choice(['spiral', 'diagonal_grid', 'random_blobs', 'fractal_like'])
                
                if pattern == 'spiral':
                    # Spiral pattern
                    for angle in range(0, 360, 15):
                        rad = np.deg2rad(angle)
                        r = angle / 360 * 10 + 2
                        x = int(14 + r * np.cos(rad))
                        y = int(14 + r * np.sin(rad))
                        if 0 <= x < 28 and 0 <= y < 28:
                            img[max(0, y-1):min(28, y+2), max(0, x-1):min(28, x+2)] = 1.0
                
                elif pattern == 'diagonal_grid':
                    # Diagonal grid
                    for i in range(0, 28, 4):
                        for j in range(28):
                            if 0 <= i+j < 28:
                                img[i+j, j] = 1.0
                            if 0 <= i-j+27 < 28:
                                img[i-j+27, j] = 1.0
                
                elif pattern == 'random_blobs':
                    # Random blobs
                    for _ in range(5):
                        cx = np.random.randint(5, 23)
                        cy = np.random.randint(5, 23)
                        r = np.random.randint(2, 5)
                        y, x = np.ogrid[:28, :28]
                        mask = (x - cx)**2 + (y - cy)**2 <= r**2
                        img[mask] = 1.0
                
                else:  # fractal_like
                    # Fractal-like recursive squares
                    def draw_square(img, x1, y1, size, depth):
                        if depth == 0 or size < 2:
                            return
                        x2, y2 = x1 + size, y1 + size
                        img[y1:y2, x1:x2] = 1.0
                        new_size = size // 2
                        draw_square(img, x1, y1, new_size, depth-1)
                        draw_square(img, x2-new_size, y2-new_size, new_size, depth-1)
                    
                    draw_square(img, 6, 6, 16, 3)
                
                stim.append(np.clip(img, 0, 1))
        
        else:
            stim = [np.random.rand(28, 28).astype(np.float32) * 0.5 + 0.25 for _ in range(n)]
        
        return stim
    
    def extract_head_activations(self, state):
        """Extract 8-head activation pattern"""
        self.extraction_stats['total_trials'] += 1
        
        try:
            # Priority 1: Extract from workspace_content.attention_maps (REAL per-head)
            if hasattr(state, 'workspace_content') and state.workspace_content is not None:
                wc = state.workspace_content
                
                if hasattr(wc, 'attention_maps') and wc.attention_maps is not None:
                    attn_maps = wc.attention_maps
                    
                    if hasattr(attn_maps, 'detach'):
                        attn_maps = attn_maps.detach().cpu().numpy()
                    
                    if len(attn_maps.shape) == 4:
                        # V7: Different aggregation strategy
                        # attn_maps shape: [B, H, L, L]
                        # Sum over query positions, average over key positions
                        head_activations = attn_maps.sum(axis=2).mean(axis=(0, 2))
                        
                        if len(head_activations) == 8:
                            self.extraction_stats['real_attention_maps'] += 1
                            return head_activations
            
            # Fallback
            if hasattr(state, 'diagnostics') and state.diagnostics:
                diag = state.diagnostics
                if 'workspace' in diag:
                    ws_info = diag['workspace']
                    if isinstance(ws_info, dict) and 'attention_weights' in ws_info:
                        attn_weights = ws_info['attention_weights']
                        if len(attn_weights) > 0:
                            base_act = float(np.mean(attn_weights))
                            self.extraction_stats['workspace_info'] += 1
                            return np.ones(8) * base_act
            
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
        print(f"Stimulus Type: {stimulus_type}")
        print(f"{'='*70}")
        
        stimuli = self.gen_stimuli(stimulus_type, n=n_trials)
        print(f"Generated {len(stimuli)} enhanced stimuli")
        
        trial_acts = []
        trial_matches = []
        trial_metrics = {'phi': [], 'fe': []}
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
                
                if hasattr(state, 'consciousness_metrics') and state.consciousness_metrics:
                    trial_metrics['phi'].append(state.consciousness_metrics.get('phi_value', 0))
                if hasattr(state, 'self_representation') and state.self_representation:
                    trial_metrics['fe'].append(state.self_representation.get('prediction_error', 0))
                
            except KeyboardInterrupt:
                print("\n  [Interrupted]")
                break
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
        
        if len(trial_metrics['phi']) > 0:
            print(f"  Avg Φ Value: {np.mean(trial_metrics['phi']):.4f}")
        
        return {
            'match_score': float(score),
            'mean_head_activations': mean_act.tolist(),
            'std_head_activations': std_act.tolist(),
            'n_trials': len(trial_acts),
            'winner_rate': float(winner_rate),
            'metrics': {k: float(np.mean(v)) if len(v) > 0 else 0 for k, v in trial_metrics.items()}
        }
    
    def run(self):
        """Run complete experiment"""
        print("\n" + "="*80)
        print("Experiment 2: Interpretability Validation (V7 - Clean & Optimized)")
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
        print("OVERALL RESULTS (V7)")
        print(f"{'='*80}")
        print(f"Total Match Score: {total_match:.2%}")
        print(f"Avg Winner State Rate: {avg_winner_rate:.1%}")
        print(f"Status: {'✅ SUCCESS' if success else '⚠️ PARTIAL' if partial else '❌ FAILED'}")
        
        print(f"\n  Extraction Statistics:")
        total = self.extraction_stats['total_trials']
        if total > 0:
            print(f"    Real Attention Maps: {self.extraction_stats['real_attention_maps']}/{total} ({100*self.extraction_stats['real_attention_maps']/total:.1f}%)")
            print(f"    Workspace Info: {self.extraction_stats['workspace_info']}/{total}")
            print(f"    Fallback: {self.extraction_stats['fallback']}/{total}")
        
        report = {
            'experiment': 'Exp-2: Interpretability Validation',
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'test_cases': list(results.keys()),
            'results': results,
            'overall_statistics': {
                'total_match_score': float(total_match),
                'avg_winner_rate': float(avg_winner_rate),
                'success': bool(success),
                'partial_success': bool(partial)
            },
            'extraction_statistics': self.extraction_stats,
            'conclusions': [
                f"V7: Optimized stimuli for head differentiation",
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
            ax.set_ylabel('Activation Strength', fontsize=11)
            ax.set_title(f"{key.replace('_', ' ').title()}\nMatch: {result['match_score']:.0%}", fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'H{i}' for i in range(8)])
            ax.set_ylim(0, max(1.0, mean_act.max() * 1.2))
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('NCT Interpretability Validation (V7)\nOptimized Stimuli', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        viz_path = self.results_dir / 'interpretability_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")


def main():
    print("="*80)
    print("Experiment 2 V7: NCT Interpretability Validation (Clean & Optimized)")
    print("V7 Improvements:")
    print("  1. CPU mode for cleaner logs (no device errors)")
    print("  2. Optimized stimuli for head differentiation")
    print("="*80)
    
    exp = Exp2V7(threshold=0.3)
    report = exp.run()
    
    print("\n" + "="*80)
    print("V7 RESULTS SUMMARY")
    print("="*80)
    print(f"Match Score: {report['overall_statistics']['total_match_score']:.2%}")
    print(f"Winner Rate: {report['overall_statistics']['avg_winner_rate']:.1f}%")


if __name__ == '__main__':
    main()
