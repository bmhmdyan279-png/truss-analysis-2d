import io
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
except AttributeError:
    pass

from truss_analysis import Element, Node, solve
from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.postprocess import calculate_element_forces

# مثال: خرپای ساده مثلثی
nodes = [
    Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    Node(id="2", x=2.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    Node(id="3", x=1.0, y=1.5, is_support=False),
]

elements = [
    Element(id="1", node_i="1", node_j="3", E=210e9, A=0.01),
    Element(id="2", node_i="2", node_j="3", E=210e9, A=0.01),
    Element(id="3", node_i="1", node_j="2", E=210e9, A=0.01),
]

K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)

# اعمال بار -10000 نیوتن در جهت Y به گره 3
F_ext[5] += -10000.0
F_mech[5] += -10000.0

print("🔍 تحلیل خرپای نمونه...")
U = solve(K, F_ext, fixed_dofs)
results, strain_energy, prestress_work = calculate_element_forces(nodes, elements, U)

print("\n📊 نتایج:")
for r in results:
    status_icon = "📈" if r["force"] > 0 else "📉"
    print(f"  عضو {r['element']}: {status_icon} N={r['force']:.2f} N")
