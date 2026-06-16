# test_super_simple.py
import numpy as np
from model import Node, Element

print("🧪 تست فوق‌ساده علامت حرارتی")

# ۱. دو گره با فاصله ۲ متر
node1 = Node(id=1, x=0, y=0, is_support=True)
node2 = Node(id=2, x=2, y=0, is_support=True)

# ۲. یک عضو بین آنها
element = Element(
    id=1,
    node_i=node1,
    node_j=node2,
    A=0.01,      # 0.01 m²
    E=210e9,     # 210 GPa
    alpha=1.2e-5,
    delta_T=100  # ۱۰۰ درجه افزایش دما
)

print(f"📏 طول عضو: {element.L} m")
print(f"🔥 ΔL_free (گرما): {element.delta_L_free} m")
print(f"📐 جهت: c={element.c}, s={element.s}")

# ۳. محاسبه دستی نیروی حرارتی
AE_L = element.A * element.E / element.L
print(f"💰 AE/L = {AE_L:.3e} N/m")

# با قرارداد جدید: f = AE/L * ΔL_free * [-c, -s, c, s]
f_manual = AE_L * element.delta_L_free * np.array([-1, 0, 1, 0])  # c=1, s=0
print(f"🎯 نیروی حرارتی دستی: {f_manual}")

# ۴. بررسی منطقی
if f_manual[0] < 0 and f_manual[2] > 0:
    print("✅ منطقی است: نیروها به سمت داخل عضو (فشاری)")
else:
    print("❌ مشکلی وجود دارد!")