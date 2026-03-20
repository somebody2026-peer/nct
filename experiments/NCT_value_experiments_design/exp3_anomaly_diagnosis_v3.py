# -*- coding: utf-8 -*-
"""
Experiment 3: Anomaly Diagnosis (V3 - Relative Metrics & Multi-feature)
=========================================================================
Exp-3: Anomaly Diagnosis - Multi-dimensional Anomaly Reasoning

V3 Key Improvements:
1. Use BASELINE metrics (first sample) for relative comparison
2. Extract head entropy from head_contributions (not attention_weights)
3. Add more diagnostic features: confidence, salience variance
4. Improved diagnosis logic using delta metrics and multi-feature patterns

V2 Issues Fixed:
- PE always high (3.3+) after first sample - fixed by using relative PE
- Attention Entropy nan - fixed by extracting from head_contributions
- Poor discrimination - fixed by multi-feature combination

Version History:
- V1: Base version, metric extraction issues
- V2: Fixed PE extraction but still poor discrimination
- V3: Relative metrics + multi-feature diagnosis

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


class Exp3V3:
    """Anomaly Diagnosis Experiment V3 - Relative Metrics & Multi-feature"""
    
    # Diagnosis rules based on multi-indicator patterns
    DIAGNOSIS_RULES = {
        'noise_corruption': {
            'description': 'Noise Corruption - Input quality severely degraded',
            'conditions': {'pe_high': True, 'phi_low': True, 'attention_dispersed': True},
            'suggestion': 'Check input data quality, possible severe noise'
        },
        'occlusion': {
            'description': 'Occlusion - Partial input missing',
            'conditions': {'pe_medium': True, 'phi_low': True, 'attention_boundary': True},
            'suggestion': 'Input may be partially blocked, check for occlusion'
        },
        'out_of_distribution': {
            'description': 'Out-of-Distribution - Unknown pattern type',
            'conditions': {'pe_high': True, 'phi_very_low': True, 'salience_abnormal': True},
            'suggestion': 'Expand training set or label as new category'
        },
        'adversarial': {
            'description': 'Adversarial Sample - Possible attack',
            'conditions': {'pe_normal': True, 'phi_very_low': True, 'attention_focused': True},
            'suggestion': 'Perform adversarial detection, possible attack'
        },
        'concept_drift': {
            'description': 'Concept Drift - Distribution shift detected',
            'conditions': {'pe_normal': True, 'phi_low': True, 'neuromodulator_abnormal': True},
            'suggestion': 'Model may need retraining for new distribution'
        }
    }
    
    def __init__(self, use_gpu=False, pretrained_path=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        print(f"Using device: {self.device}")
        
        config = NCTConfig(
            n_heads=8,
            n_layers=4,
            d_model=512,
            dim_ff=2048,
            visual_patch_size=4,
            visual_embed_dim=256,
            consciousness_threshold=0.3
        )
        
        self.nct = NCTManager(config)
        self.nct.to(self.device)
        self.nct.eval()
        
        # Load pretrained model if available
        self.model_type = "random_init"
        if pretrained_path and Path(pretrained_path).exists():
            try:
                checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.nct.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    self.model_type = "pretrained"
                    print(f"✅ Loaded pretrained model from: {pretrained_path}")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained model: {e}")
        
        self.version = f"v3-RelativeMetrics-{self.model_type}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path('results') / f'exp3_anomaly_diagnosis_{self.version}_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {self.results_dir}")
    
    def generate_anomaly(self, anomaly_type, base_image=None, n_samples=20):
        """Generate anomalous samples of specified type
        
        Args:
            anomaly_type: Type of anomaly to generate
            base_image: Base image to corrupt (if None, generate random)
            n_samples: Number of samples to generate
        
        Returns:
            List of anomalous images
        """
        samples = []
        
        for _ in range(n_samples):
            if base_image is not None:
                img = base_image.copy()
            else:
                # Generate base image (simple pattern)
                img = np.zeros((28, 28), dtype=np.float32)
                # Add some structure
                cx, cy = np.random.randint(8, 20, 2)
                y, x = np.ogrid[:28, :28]
                mask = (x - cx)**2 + (y - cy)**2 <= 6**2
                img[mask] = 1.0
                # Add some lines
                for _ in range(2):
                    x1, x2 = np.random.randint(4, 24, 2)
                    y_pos = np.random.randint(4, 24)
                    img[y_pos-1:y_pos+1, x1:x2] = 1.0
            
            if anomaly_type == 'noise_corruption':
                # Add heavy Gaussian noise
                noise_level = np.random.uniform(0.5, 1.0)
                noise = np.random.randn(28, 28).astype(np.float32) * noise_level
                img = np.clip(img + noise, 0, 1)
            
            elif anomaly_type == 'occlusion':
                # Random occlusion
                occ_type = np.random.choice(['block', 'stripe', 'random'])
                if occ_type == 'block':
                    x1, y1 = np.random.randint(4, 14, 2)
                    x2, y2 = x1 + np.random.randint(8, 14), y1 + np.random.randint(8, 14)
                    img[y1:y2, x1:x2] = 0.0
                elif occ_type == 'stripe':
                    for i in range(3):
                        y = np.random.randint(4, 24)
                        img[y:y+4, :] = 0.0
                else:
                    mask = np.random.rand(28, 28) > 0.7
                    img[mask] = 0.0
            
            elif anomaly_type == 'out_of_distribution':
                # Generate completely different patterns
                pattern = np.random.choice(['checkerboard', 'diagonal', 'spiral', 'random'])
                img = np.zeros((28, 28), dtype=np.float32)
                if pattern == 'checkerboard':
                    for i in range(0, 28, 4):
                        for j in range(0, 28, 4):
                            if (i + j) % 8 == 0:
                                img[i:i+4, j:j+4] = 1.0
                elif pattern == 'diagonal':
                    for i in range(28):
                        for j in range(28):
                            if (i + j) % 4 < 2:
                                img[i, j] = 1.0
                elif pattern == 'spiral':
                    for angle in range(0, 360, 10):
                        rad = np.deg2rad(angle)
                        r = angle / 360 * 12 + 2
                        x = int(14 + r * np.cos(rad))
                        y = int(14 + r * np.sin(rad))
                        if 0 <= x < 28 and 0 <= y < 28:
                            img[max(0,y-1):min(28,y+2), max(0,x-1):min(28,x+2)] = 1.0
                else:
                    img = np.random.rand(28, 28).astype(np.float32)
            
            elif anomaly_type == 'adversarial':
                # Simulate adversarial perturbation
                # Small targeted perturbations that don't change visual appearance much
                # but are designed to fool models
                perturbation = np.random.randn(28, 28).astype(np.float32) * 0.1
                # Add structured perturbation
                for _ in range(5):
                    x, y = np.random.randint(0, 28, 2)
                    perturbation[max(0,y-2):min(28,y+3), max(0,x-2):min(28,x+3)] += 0.3
                img = np.clip(img + perturbation, 0, 1)
            
            elif anomaly_type == 'concept_drift':
                # Rotated or scaled versions
                transform = np.random.choice(['rotate', 'scale', 'invert'])
                if transform == 'rotate':
                    from scipy.ndimage import rotate
                    angle = np.random.uniform(30, 60)
                    img = rotate(img, angle, reshape=False, mode='constant')
                    img = np.clip(img, 0, 1).astype(np.float32)
                elif transform == 'scale':
                    from scipy.ndimage import zoom
                    scale = np.random.uniform(0.5, 0.8)
                    scaled = zoom(img, scale)
                    pad = (28 - scaled.shape[0]) // 2
                    img = np.zeros((28, 28), dtype=np.float32)
                    img[pad:pad+scaled.shape[0], pad:pad+scaled.shape[1]] = scaled
                else:
                    img = 1.0 - img
            
            samples.append(img)
        
        return samples
    
    def extract_diagnosis_metrics(self, state):
        """Extract multi-dimensional diagnosis metrics from NCT state
        
        V3 Improvements:
        1. Extract head entropy from head_contributions (more reliable)
        2. Add salience variance for pattern discrimination
        3. Track all available features for diagnosis
        """
        metrics = {
            'pe': 0.0,
            'phi': 0.0,
            'attention_entropy': 0.0,
            'salience': 0.0,
            'confidence': 0.0,
            'consciousness_level': 'UNKNOWN',
            'head_entropies': [],
            'salience_variance': 0.0
        }
        
        try:
            # Extract from consciousness_metrics
            if hasattr(state, 'consciousness_metrics') and state.consciousness_metrics:
                cm = state.consciousness_metrics
                metrics['phi'] = float(cm.get('phi_value', 0.0))
                metrics['consciousness_level'] = cm.get('consciousness_level', 'UNKNOWN')
            
            # Extract from self_representation
            if hasattr(state, 'self_representation') and state.self_representation:
                sr = state.self_representation
                metrics['pe'] = float(sr.get('free_energy', 0.0))
                metrics['confidence'] = float(sr.get('confidence', 0.0))
            
            # Extract from diagnostics (primary source for attention and salience)
            if hasattr(state, 'diagnostics') and state.diagnostics:
                diag = state.diagnostics
                
                # Get workspace info
                if 'workspace' in diag:
                    ws = diag['workspace']
                    
                    # Salience
                    if 'winner_salience' in ws:
                        metrics['salience'] = float(ws['winner_salience'])
                    
                    # V3: Extract entropy from head_contributions (more detailed)
                    if 'head_contributions' in ws:
                        hc = ws['head_contributions']
                        if isinstance(hc, dict):
                            entropies = []
                            for head_name in sorted(hc.keys()):
                                if isinstance(hc[head_name], dict) and 'entropy' in hc[head_name]:
                                    entropies.append(float(hc[head_name]['entropy']))
                            if entropies:
                                metrics['head_entropies'] = entropies
                                metrics['attention_entropy'] = float(np.mean(entropies))
                    
                    # V3: Calculate salience variance across candidates
                    if 'all_candidates_salience' in ws:
                        acs = ws['all_candidates_salience']
                        if isinstance(acs, (list, np.ndarray)) and len(acs) > 1:
                            metrics['salience_variance'] = float(np.var(acs))
            
        except Exception as e:
            print(f"  [Warning] Metric extraction error: {e}")
        
        return metrics
    
    def diagnose(self, metrics, baseline_metrics=None):
        """Generate diagnosis based on multi-indicator pattern matching
        
        V3: Use RELATIVE metrics (delta from baseline) and multi-feature patterns
        """
        pe = metrics['pe']
        phi = metrics['phi']
        attn_entropy = metrics['attention_entropy'] or 0.65
        salience = metrics['salience']
        confidence = metrics['confidence']
        salience_var = metrics.get('salience_variance', 0.0)
        head_entropies = metrics.get('head_entropies', [])
        
        # V3: Calculate relative metrics (delta from baseline)
        if baseline_metrics:
            delta_pe = pe - baseline_metrics['pe']
            delta_phi = phi - baseline_metrics['phi']
            delta_entropy = attn_entropy - baseline_metrics['attention_entropy']
            delta_salience = salience - baseline_metrics['salience']
            delta_confidence = confidence - baseline_metrics['confidence']
        else:
            delta_pe = 0
            delta_phi = 0
            delta_entropy = 0
            delta_salience = 0
            delta_confidence = 0
        
        diagnosis = {
            'type': 'unknown',
            'description': 'Unknown anomaly type',
            'confidence': 0.0,
            'indicators': {},
            'suggestion': 'Unable to diagnose, manual inspection recommended'
        }
        
        # V3: Multi-feature pattern matching with relative metrics
        # Pattern 1: Noise Corruption - PE↑↑↑, Φ↓, Entropy↑ (dispersed), Confidence↓↓
        if delta_pe > 2.5 and delta_phi < -0.05 and delta_entropy > 0.02 and delta_confidence < -0.1:
            diagnosis['type'] = 'noise_corruption'
            diagnosis['description'] = 'Noise Corruption - Input quality severely degraded'
            diagnosis['confidence'] = min(0.9, 0.6 + abs(delta_pe) * 0.1)
            diagnosis['suggestion'] = 'Check input data quality, possible severe noise'
        
        # Pattern 2: Occlusion - PE↑↑, Φ↓, Salience variance↑ (focused on boundary)
        elif delta_pe > 1.5 and delta_phi < -0.03 and salience_var > 0.02:
            diagnosis['type'] = 'occlusion'
            diagnosis['description'] = 'Occlusion - Partial input missing'
            diagnosis['confidence'] = min(0.85, 0.5 + salience_var * 2)
            diagnosis['suggestion'] = 'Input may be partially blocked, check for occlusion'
        
        # Pattern 3: Out-of-Distribution - PE↑↑, Φ↓↓, All features abnormal
        elif delta_pe > 2.0 and delta_phi < -0.1 and len(head_entropies) > 0:
            avg_head_entropy = np.mean(head_entropies)
            if avg_head_entropy > 0.65 or avg_head_entropy < 0.55:
                diagnosis['type'] = 'out_of_distribution'
                diagnosis['description'] = 'Out-of-Distribution - Unknown pattern type'
                diagnosis['confidence'] = min(0.9, 0.6 + abs(delta_phi) * 2)
                diagnosis['suggestion'] = 'Expand training set or label as new category'
        
        # Pattern 4: Adversarial - PE normal/slight↑, Φ↓↓↓, Attention abnormally focused
        elif delta_pe < 2.0 and delta_phi < -0.15 and attn_entropy < 0.60:
            diagnosis['type'] = 'adversarial'
            diagnosis['description'] = 'Adversarial Sample - Possible attack'
            diagnosis['confidence'] = min(0.85, 0.6 + abs(delta_phi) * 3)
            diagnosis['suggestion'] = 'Perform adversarial detection, possible attack'
        
        # Pattern 5: Concept Drift - PE normal, Φ↓, Confidence↓
        elif abs(delta_pe) < 1.5 and delta_phi < -0.05 and delta_confidence < -0.05:
            diagnosis['type'] = 'concept_drift'
            diagnosis['description'] = 'Concept Drift - Distribution shift detected'
            diagnosis['confidence'] = min(0.75, 0.5 + abs(delta_phi) * 2)
            diagnosis['suggestion'] = 'Model may need retraining for new distribution'
        
        # Record indicator status
        diagnosis['indicators'] = {
            'delta_pe': f"{delta_pe:+.3f}",
            'delta_phi': f"{delta_phi:+.3f}",
            'delta_entropy': f"{delta_entropy:+.3f}",
            'delta_salience': f"{delta_salience:+.3f}",
            'delta_confidence': f"{delta_confidence:+.3f}",
            'salience_variance': f"{salience_var:.4f}",
            'avg_head_entropy': f"{np.mean(head_entropies):.3f}" if head_entropies else "N/A"
        }
        
        return diagnosis
    
    def run_single_type(self, anomaly_type, n_samples=20):
        """Run diagnosis for a single anomaly type
        
        V3: Use BASELINE from first sample for relative comparison
        """
        print(f"\n{'='*70}")
        print(f"Anomaly Type: {anomaly_type}")
        print(f"{'='*70}")
        
        samples = self.generate_anomaly(anomaly_type, n_samples=n_samples)
        print(f"Generated {len(samples)} anomalous samples")
        
        all_metrics = []
        all_diagnoses = []
        baseline_metrics = None  # V3: Track baseline
        
        for i, sample in enumerate(samples):
            if i % 5 == 0:
                print(f"  Processing {i}/{len(samples)}...")
            
            try:
                state = self.nct.process_cycle({'visual': sample})
                metrics = self.extract_diagnosis_metrics(state)
                
                # V3: Set baseline from first sample
                if i == 0:
                    baseline_metrics = metrics.copy()
                    print(f"  [BASELINE] PE={metrics['pe']:.3f}, Φ={metrics['phi']:.3f}, Entropy={metrics['attention_entropy']:.3f}")
                
                # V3: Pass baseline to diagnose
                diagnosis = self.diagnose(metrics, baseline_metrics)
                
                all_metrics.append(metrics)
                all_diagnoses.append(diagnosis)
                
            except Exception as e:
                print(f"  [Error] Sample {i}: {e}")
        
        # Aggregate results
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics if isinstance(m[k], (int, float))])
            for k in ['pe', 'phi', 'attention_entropy', 'salience', 'confidence']
        }
        
        # Calculate diagnosis accuracy
        correct_diagnoses = sum(1 for d in all_diagnoses if d['type'] == anomaly_type)
        accuracy = correct_diagnoses / len(all_diagnoses) if all_diagnoses else 0
        
        print(f"\n  Results:")
        print(f"  Diagnosis Accuracy: {accuracy:.2%}")
        print(f"  Average Metrics:")
        print(f"    PE (Prediction Error): {avg_metrics['pe']:.4f}")
        print(f"    Φ (Phi Value): {avg_metrics['phi']:.4f}")
        print(f"    Attention Entropy: {avg_metrics['attention_entropy']:.4f}")
        print(f"    Salience: {avg_metrics['salience']:.4f}")
        
        return {
            'anomaly_type': anomaly_type,
            'n_samples': len(samples),
            'avg_metrics': avg_metrics,
            'diagnosis_accuracy': float(accuracy),
            'all_metrics': all_metrics,
            'all_diagnoses': all_diagnoses
        }
    
    def run(self):
        """Run complete experiment"""
        print("\n" + "="*80)
        print("Experiment 3: Anomaly Diagnosis (V3 - Relative Metrics & Multi-feature)")
        print("V3: Use baseline comparison and multi-feature patterns")
        print("="*80)
        
        anomaly_types = [
            'noise_corruption',
            'occlusion', 
            'out_of_distribution',
            'adversarial',
            'concept_drift'
        ]
        
        results = {}
        for atype in anomaly_types:
            results[atype] = self.run_single_type(atype, n_samples=20)
        
        # Calculate overall statistics
        overall_accuracy = np.mean([r['diagnosis_accuracy'] for r in results.values()])
        
        print(f"\n{'='*80}")
        print("OVERALL RESULTS (V3)")
        print(f"{'='*80}")
        print(f"Overall Diagnosis Accuracy: {overall_accuracy:.2%}")
        print(f"\nPer-Type Accuracy:")
        for atype, r in results.items():
            status = "✅" if r['diagnosis_accuracy'] > 0.5 else "❌"
            print(f"  {status} {atype}: {r['diagnosis_accuracy']:.2%}")
        
        # Save report
        report = {
            'experiment': 'Exp-3: Anomaly Diagnosis',
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'anomaly_types': anomaly_types,
            'results': {k: {
                'diagnosis_accuracy': v['diagnosis_accuracy'],
                'avg_metrics': v['avg_metrics'],
                'n_samples': v['n_samples']
            } for k, v in results.items()},
            'overall_statistics': {
                'overall_accuracy': float(overall_accuracy)
            },
            'conclusions': [
                f"V3: Relative metrics + multi-feature diagnosis",
                f"Overall diagnosis accuracy: {overall_accuracy:.2%}"
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
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        anomaly_types = list(results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # Plot 1: Diagnosis Accuracy
        ax = axes[0]
        accuracies = [results[atype]['diagnosis_accuracy'] for atype in anomaly_types]
        bars = ax.bar(range(len(anomaly_types)), accuracies, color=colors)
        ax.set_xticks(range(len(anomaly_types)))
        ax.set_xticklabels([atype.replace('_', '\n') for atype in anomaly_types], fontsize=8)
        ax.set_ylabel('Diagnosis Accuracy')
        ax.set_title('Diagnosis Accuracy by Anomaly Type', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='red', linestyle='--', label='Chance Level')
        ax.legend()
        
        # Plots 2-5: Metrics for each anomaly type
        metrics_names = ['pe', 'phi', 'attention_entropy', 'salience']
        metrics_labels = ['Prediction Error (PE)', 'Φ Value', 'Attention Entropy', 'Salience']
        
        for idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
            ax = axes[idx + 1]
            values = [results[atype]['avg_metrics'][metric] for atype in anomaly_types]
            ax.bar(range(len(anomaly_types)), values, color=colors)
            ax.set_xticks(range(len(anomaly_types)))
            ax.set_xticklabels([atype.replace('_', '\n') for atype in anomaly_types], fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(f'{label} by Anomaly Type', fontweight='bold')
        
        # Plot 6: Summary
        ax = axes[5]
        ax.axis('off')
        summary_text = "Experiment 3: Anomaly Diagnosis\n\n"
        summary_text += "Key Findings:\n"
        for atype in anomaly_types:
            r = results[atype]
            summary_text += f"\n• {atype}:\n"
            summary_text += f"  Accuracy: {r['diagnosis_accuracy']:.1%}\n"
            summary_text += f"  PE={r['avg_metrics']['pe']:.3f}, Φ={r['avg_metrics']['phi']:.3f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('NCT Anomaly Diagnosis Experiment (V1)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        viz_path = self.results_dir / 'anomaly_diagnosis_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")


def main():
    print("="*80)
    print("Experiment 3 V3: NCT Anomaly Diagnosis (Relative Metrics)")
    print("V3: Baseline comparison + multi-feature patterns")
    print("="*80)
    
    # Try to find pretrained model - use V3 model
    project_root = Path(__file__).parent.parent.parent
    pretrained_paths = [
        project_root / "results/training_v3/best_model_v3.pt",  # NCT V3 model
        project_root / "results/anomaly_detection_complete_20260227_005219/best_model.pt",
    ]
    
    pretrained_path = None
    for path in pretrained_paths:
        if path.exists():
            pretrained_path = str(path)
            print(f"Found pretrained model: {path}")
            break
    
    exp = Exp3V3(use_gpu=False, pretrained_path=pretrained_path)
    report = exp.run()
    
    print("\n" + "="*80)
    print("V3 RESULTS SUMMARY")
    print("="*80)
    print(f"Overall Accuracy: {report['overall_statistics']['overall_accuracy']:.2%}")


if __name__ == '__main__':
    main()
