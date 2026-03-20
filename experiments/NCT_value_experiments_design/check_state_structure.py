# Check NCT state structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nct_modules import NCTManager, NCTConfig
import numpy as np

config = NCTConfig(consciousness_threshold=0.3)
nct = NCTManager(config)
nct.eval()

# Process a sample
img = np.random.rand(28, 28).astype(np.float32)
state = nct.process_cycle({'visual': img})

print('=== State Attributes ===')
for attr in dir(state):
    if not attr.startswith('_'):
        print(f'  {attr}')

print('\n=== self_representation ===')
if hasattr(state, 'self_representation'):
    sr = state.self_representation
    print(f'  Type: {type(sr)}')
    if sr:
        print(f'  Content: {sr}')

print('\n=== consciousness_metrics ===')
if hasattr(state, 'consciousness_metrics'):
    cm = state.consciousness_metrics
    print(f'  Type: {type(cm)}')
    if cm:
        print(f'  Content: {cm}')

print('\n=== diagnostics ===')
if hasattr(state, 'diagnostics'):
    diag = state.diagnostics
    print(f'  Type: {type(diag)}')
    if diag:
        print(f'  Keys: {list(diag.keys()) if isinstance(diag, dict) else "N/A"}')
        if 'workspace' in diag:
            ws = diag['workspace']
            print(f'  workspace keys: {list(ws.keys()) if isinstance(ws, dict) else "N/A"}')
            if 'attention_weights' in ws:
                aw = ws['attention_weights']
                print(f'  attention_weights type: {type(aw)}')
                print(f'  attention_weights len: {len(aw) if hasattr(aw, "__len__") else "N/A"}')
                print(f'  attention_weights: {aw}')
            if 'head_contributions' in ws:
                hc = ws['head_contributions']
                print(f'  head_contributions type: {type(hc)}')
                print(f'  head_contributions: {hc}')
            if 'all_candidates_salience' in ws:
                acs = ws['all_candidates_salience']
                print(f'  all_candidates_salience: {acs}')
        if 'prediction_error' in diag:
            print(f'  prediction_error: {diag["prediction_error"]}')
