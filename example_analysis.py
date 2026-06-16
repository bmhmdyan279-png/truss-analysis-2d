import sys
sys.path.append('.')

from model import TrussModel
from assembly import build_global_matrices
from solver import solve_displacements, calculate_element_results
from postprocess import generate_report

# مثال: خرپای ساده مثلثی
input_data = {
    'units': 'SI',
    'nodes': [
        {'id': 1, 'x': 0, 'y': 0, 'is_support': True},
        {'id': 2, 'x': 2, 'y': 0, 'is_support': True},
        {'id': 3, 'x': 1, 'y': 1.5, 'is_support': False}
    ],
    'elements': [
        {'id': 1, 'node_i': 1, 'node_j': 3, 'A': 0.01, 'E': 210e9},
        {'id': 2, 'node_i': 2, 'node_j': 3, 'A': 0.01, 'E': 210e9},
        {'id': 3, 'node_i': 1, 'node_j': 2, 'A': 0.01, 'E': 210e9}
    ],
    'loads': {
        'node_forces': [
            {'node_id': 3, 'Fx': 0, 'Fy': -10000}
        ]
    },
    'options': {
        'use_sparse': True,
        'bc_method': 'elimination'
    }
}

print("🔍 تحلیل خرپای نمونه...")
truss = TrussModel(input_data)
K, F = build_global_matrices(truss)
displacements = solve_displacements(truss, K, F)
results = calculate_element_results(truss, displacements)

print("\n📊 نتایج:")
for r in results:
    status_icon = "📈" if r['status'] == 'Tension' else "📉"
    print(f"  عضو {r['id']}: {status_icon} N={r['N']:.2f} N ({r['status']})")