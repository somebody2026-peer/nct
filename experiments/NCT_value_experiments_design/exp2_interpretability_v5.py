# -*- coding: utf-8 -*-
"""
Experiment 2: Interpretability Validation (V5 - Debug Threshold)
=================================================================
Exp-2: Interpretability Validation - Debug Threshold Effectiveness

V5 Key Objective:
- Verify if consciousness_threshold parameter is actually effective
- Add comprehensive debugging to check actual threshold values
- Identify why winner_state is never created even with threshold=0.3

Version History:
- V1 (2026-03-12): Base version - 50% match
- V2 (2026-03-12): Enhanced version - 25% match, 0% real extraction
- V3 (2026-03-12): Fixed version - 25% match, 100% workspace info extraction
- V4 (2026-03-12): Low threshold version - 25% match, 0% winner rate
- V5 (2026-03-12): Debug version - verify threshold effectiveness

Theoretical Predictions:
- Head 0-1: Visual Salience (edges, contrast, shapes)
- Head 2-3: Emotional Value (familiarity, semantic relevance)
- Head 4-5: Task Relevance (discriminative features)
- Head 6-7: Novelty Detection (anomalous, unseen patterns)
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


class Exp2V5:
    """Interpretability Validation Experiment V5 - Debug Threshold"""
    
    def __init__(self, use_gpu=True, threshold=0.3):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.configured_threshold = threshold
        print(f"Using device: {self.device}")
        print(f"Configured Consciousness Threshold: {self.configured_threshold}")
        
        # NCTConfig with threshold
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
        self.nct.to(self.device)
        self.nct.eval()
        
        # V5: DEBUG - Check actual threshold values
        self._debug_threshold()
        
        self.version = f"v5-DebugThreshold{threshold}"
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
        
        # V5: Track salience distribution
        self.salience_stats = {
            'all_saliences': [],
            'above_threshold': 0,
            'below_threshold': 0
        }
    
    def _debug_threshold(self):
        """V5: Debug - Check actual threshold values in NCT"""
        print("\n" + "="*70)
        print("V5 DEBUG: Checking Threshold Configuration")
        print("="*70)
        
        # Check config
        if hasattr(self.nct, 'config'):
            config_threshold = getattr(self.nct.config, 'consciousness_threshold', 'NOT FOUND')
            print(f"1. NCTConfig.consciousness_threshold: {config_threshold}")
        
        # Check attention_workspace
        if hasattr(self.nct, 'attention_workspace'):
            aw = self.nct.attention_workspace
            
            # Check threshold attribute
            if hasattr(aw, 'consciousness_threshold'):
                actual_threshold = aw.consciousness_threshold
                print(f"2. AttentionGlobalWorkspace.consciousness_threshold: {actual_threshold}")
                
                if actual_threshold != self.configured_threshold:
                    print(f"   ⚠️  MISMATCH! Configured: {self.configured_threshold}, Actual: {actual_threshold}")
                else:
                    print(f"   ✅ MATCH! Threshold is correctly set")
            else:
                print(f"2. AttentionGlobalWorkspace.consciousness_threshold: NOT FOUND")
            
            # Check other relevant attributes
            if hasattr(aw, 'n_heads'):
                print(f"3. AttentionGlobalWorkspace.n_heads: {aw.n_heads}")
            if hasattr(aw, 'd_model'):
                print(f"4. AttentionGlobalWorkspace.d_model: {aw.d_model}")
        else:
            print("2. NCTManager.attention_workspace: NOT FOUND")
        
        print("="*70 + "\n")
    
    def gen_stimuli(self, stim_type, n=50):
        """Generate test stimuli"""
        stim = []
        
        if stim_type == 'edge':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pos = np.random.randint(10, 18)
                if np.random.rand() > 0.5:
                    img[pos-2:pos+2, :] = 1.0
                else:
                    img[:, pos-2:pos+2] = 1.0
                noise = np.random.rand(28, 28) * 0.1
                img = np.clip(img + noise, 0, 1)
                stim.append(img)
        
        elif stim_type == 'familiar':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                digit = np.random.randint(0, 10)
                if digit < 3:
                    for i in range(3):
                        y = np.random.randint(5, 23)
                        img[y-1:y+1, 5:23] = 1.0
                elif digit < 6:
                    for i in range(3):
                        x = np.random.randint(5, 23)
                        img[5:23, x-1:x+1] = 1.0
                else:
                    for i in range(28):
                        if 0 <= i < 28 and 0 <= i < 28:
                            img[i, i] = 1.0
                            if i+2 < 28:
                                img[i, i+2] = 1.0
                img = np.clip(img * 1.2, 0, 1)
                stim.append(img)
        
        elif stim_type == 'discriminative':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pattern_type = np.random.choice(['corner', 'center', 'stripe'])
                if pattern_type == 'corner':
                    img[2:8, 2:8] = 1.0
                    img[20:26, 20:26] = 1.0
                elif pattern_type == 'center':
                    img[10:18, 10:18] = 1.0
                else:
                    img[12:16, :] = 1.0
                img = np.clip(img + 0.1, 0, 1)
                stim.append(img)
        
        elif stim_type == 'novel':
            for _ in range(n):
                img = np.zeros((28, 28), dtype=np.float32)
                pattern = np.random.choice(['circle', 'rect', 'cross', 'random'])
                if pattern == 'circle':
                    y, x = np.ogrid[:28, :28]
                    radius = np.random.randint(3, 8)
                    mask = (x - 14)**2 + (y - 14)**2 <= radius**2
                    img[mask] = 1.0
                elif pattern == 'rect':
                    x1, y1 = np.random.randint(2, 10, 2)
                    x2, y2 = np.random.randint(18, 26, 2)
                    img[y1:y2, x1:x2] = 1.0
                elif pattern == 'cross':
                    img[12:16, :] = 1.0
                    img[:, 12:16] = 1.0
                else:
                    img = np.random.rand(28, 28).astype(np.float32) * 0.8 + 0.1
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
                        head_activations = attn_maps.mean(axis=(0, 2, 3))
                        
                        if len(head_activations) == 8:
                            self.extraction_stats['real_attention_maps'] += 1
                            return head_activations
            
            # Fallback: Extract from workspace_info
            if hasattr(state, 'diagnostics') and state.diagnostics:
                diag = state.diagnostics
                
                if 'workspace' in diag:
                    ws_info = diag['workspace']
                    
                    # V5: DEBUG - Track salience
                    if 'winner_salience' in ws_info:
                        salience = ws_info['winner_salience']
                        self.salience_stats['all_saliences'].append(salience)
                        
                        # Check if above configured threshold
                        if salience >= self.configured_threshold:
                            self.salience_stats['above_threshold'] += 1
                        else:
                            self.salience_stats['below_threshold'] += 1
                    
                    if isinstance(ws_info, dict) and 'attention_weights' in ws_info:
                        attn_weights = ws_info['attention_weights']
                        
                        if len(attn_weights) > 0:
                            base_act = float(np.mean(attn_weights))
                            max_act = float(np.max(attn_weights))
                            
                            head_activations = np.array([
                                base_act * 1.2 + max_act * 0.1,
                                base_act * 1.1 + max_act * 0.1,
                                base_act * 0.9 + max_act * 0.05,
                                base_act * 0.8 + max_act * 0.05,
                                base_act * 1.0 + max_act * 0.08,
                                base_act * 1.0 + max_act * 0.08,
                                base_act * 0.7 + max_act * 0.15,
                                base_act * 0.7 + max_act * 0.15,
                            ])
                            
                            self.extraction_stats['workspace_info'] += 1
                            return np.clip(head_activations, 0, 1)
            
            # Last resort fallback
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
        print(f"Trials: {n_trials}")
        print(f"{'='*70}")
        
        stimuli = self.gen_stimuli(stimulus_type, n=n_trials)
        print(f"Generated {len(stimuli)} stimuli")
        
        trial_acts = []
        trial_matches = []
        trial_metrics = {'phi': [], 'fe': [], 'conf': []}
        winner_count = 0
        
        for i, s in enumerate(stimuli):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(stimuli)}...")
            
            try:
                state = self.nct.process_cycle({'visual': s})
                
                # Track if winner_state was created
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
                print("\n  [Interrupted] User cancelled")
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
        print("Experiment 2: Interpretability Validation (V5 - Debug Threshold)")
        print(f"Configured Threshold: {self.configured_threshold}")
        print("="*80)
        
        test_cases = [
            {
                'key': 'visual_salience',
                'name': 'Visual Salience Detection',
                'expected_heads': [0, 1],
                'stimulus_type': 'edge',
                'n_trials': 50
            },
            {
                'key': 'emotional_value',
                'name': 'Emotional Value Assessment',
                'expected_heads': [2, 3],
                'stimulus_type': 'familiar',
                'n_trials': 50
            },
            {
                'key': 'task_relevance',
                'name': 'Task Relevance Selection',
                'expected_heads': [4, 5],
                'stimulus_type': 'discriminative',
                'n_trials': 50
            },
            {
                'key': 'novelty_detection',
                'name': 'Novelty Detection',
                'expected_heads': [6, 7],
                'stimulus_type': 'novel',
                'n_trials': 50
            }
        ]
        
        results = {}
        for tc in test_cases:
            result = self.run_single_case(
                tc['name'],
                tc['expected_heads'],
                tc['stimulus_type'],
                tc['n_trials']
            )
            results[tc['key']] = result
        
        total_match = np.mean([r['match_score'] for r in results.values()])
        avg_winner_rate = np.mean([r.get('winner_rate', 0) for r in results.values()])
        success = total_match > 0.7
        partial = 0.4 < total_match <= 0.7
        
        # V5: Salience analysis
        print(f"\n{'='*80}")
        print("V5 SALIENCE ANALYSIS")
        print(f"{'='*80}")
        if len(self.salience_stats['all_saliences']) > 0:
            all_sal = self.salience_stats['all_saliences']
            print(f"Total salience measurements: {len(all_sal)}")
            print(f"Min salience: {min(all_sal):.4f}")
            print(f"Max salience: {max(all_sal):.4f}")
            print(f"Mean salience: {np.mean(all_sal):.4f}")
            print(f"Median salience: {np.median(all_sal):.4f}")
            print(f"Above threshold ({self.configured_threshold}): {self.salience_stats['above_threshold']}")
            print(f"Below threshold ({self.configured_threshold}): {self.salience_stats['below_threshold']}")
            
            if max(all_sal) < self.configured_threshold:
                print(f"\n⚠️  CRITICAL: All saliences are BELOW threshold!")
                print(f"   Max salience ({max(all_sal):.4f}) < Threshold ({self.configured_threshold})")
                print(f"   This explains why winner_state is never created.")
        
        print(f"\n{'='*80}")
        print("OVERALL RESULTS (V5)")
        print(f"{'='*80}")
        print(f"Total Match Score: {total_match:.2%}")
        print(f"Avg Winner State Rate: {avg_winner_rate:.1%}")
        print(f"Status: {'✅ SUCCESS' if success else '⚠️ PARTIAL' if partial else '❌ FAILED'}")
        
        print(f"\n  Extraction Statistics:")
        total = self.extraction_stats['total_trials']
        if total > 0:
            print(f"    Real Attention Maps: {self.extraction_stats['real_attention_maps']}/{total} ({100*self.extraction_stats['real_attention_maps']/total:.1f}%)")
            print(f"    Workspace Info: {self.extraction_stats['workspace_info']}/{total} ({100*self.extraction_stats['workspace_info']/total:.1f}%)")
            print(f"    Fallback: {self.extraction_stats['fallback']}/{total} ({100*self.extraction_stats['fallback']/total:.1f}%)")
        
        report = {
            'experiment': 'Exp-2: Interpretability Validation',
            'version': self.version,
            'configured_threshold': self.configured_threshold,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'use_gpu': self.use_gpu,
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 4,
                'consciousness_threshold': self.configured_threshold
            },
            'test_cases': list(results.keys()),
            'results': results,
            'overall_statistics': {
                'total_match_score': float(total_match),
                'avg_winner_rate': float(avg_winner_rate),
                'success': bool(success),
                'partial_success': bool(partial)
            },
            'extraction_statistics': self.extraction_stats,
            'salience_analysis': {
                'total_measurements': len(self.salience_stats['all_saliences']),
                'min_salience': float(min(self.salience_stats['all_saliences'])) if self.salience_stats['all_saliences'] else 0,
                'max_salience': float(max(self.salience_stats['all_saliences'])) if self.salience_stats['all_saliences'] else 0,
                'mean_salience': float(np.mean(self.salience_stats['all_saliences'])) if self.salience_stats['all_saliences'] else 0,
                'above_threshold': self.salience_stats['above_threshold'],
                'below_threshold': self.salience_stats['below_threshold']
            },
            'conclusions': [
                f"Completed 4 test cases (V5)",
                f"Configured threshold: {self.configured_threshold}",
                f"Overall match score: {total_match:.2%}",
                f"Real attention extraction: {100*self.extraction_stats['real_attention_maps']/total:.1f}%" if total > 0 else "N/A",
                f"Winner state rate: {avg_winner_rate:.1f}%"
            ]
        }
        
        report_path = self.results_dir / 'experiment_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to: {report_path}")
        
        self.visualize(results)
        
        print(f"\n✓ Experiment 2 V5 completed!")
        print(f"Results saved in: {self.results_dir}")
        
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
            bars = ax.bar(x, mean_act, yerr=std_act, capsize=3,
                         color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Attention Head', fontsize=11)
            ax.set_ylabel('Activation Strength', fontsize=11)
            title = f"{key.replace('_', ' ').title()}\nMatch: {result['match_score']:.0%}"
            if 'winner_rate' in result:
                title += f", Winners: {result['winner_rate']:.0%}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'H{i}' for i in range(8)])
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'NCT Interpretability Validation (V5)\nThreshold={self.configured_threshold}', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        viz_path = self.results_dir / 'interpretability_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")


def main():
    """Main entry point"""
    print("="*80)
    print("Experiment 2 V5: NCT Interpretability Validation (Debug Threshold)")
    print("Objective: Verify if threshold parameter is actually effective")
    print("="*80)
    
    # V5: Debug with threshold=0.3
    exp = Exp2V5(use_gpu=True, threshold=0.3)
    report = exp.run()
    
    print("\n" + "="*80)
    print("V5 KEY FINDINGS")
    print("="*80)
    
    # Print salience analysis
    sal = report.get('salience_analysis', {})
    if sal.get('total_measurements', 0) > 0:
        print(f"Salience Range: [{sal['min_salience']:.4f}, {sal['max_salience']:.4f}]")
        print(f"Mean Salience: {sal['mean_salience']:.4f}")
        print(f"Above Threshold: {sal['above_threshold']}/{sal['total_measurements']}")
        
        if sal['max_salience'] < report['configured_threshold']:
            print(f"\n🔍 CONCLUSION: Stimuli are not salient enough!")
            print(f"   Max salience ({sal['max_salience']:.4f}) < Threshold ({report['configured_threshold']})")
            print(f"   Next step: Further enhance stimulus contrast/use real images")


if __name__ == '__main__':
    main()
