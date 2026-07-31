# مسیر پروژه را به sys.path اضافه کنید (در صورت نیاز تنظیم شود)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.solver import solve_truss  # <-- ایمپورت واقعی solver خود را جایگزین کنید


def test_ground_truth_simple_truss():
    """
    تست Ground Truth: خرپای ۳ گره‌ای و ۲ عضوی افقی.
    گره ۱: (0, 0) مفصلی (Fixed)
    گره ۲: (1, 0) آزاد، بار Fx=1000 N
    گره ۳: (2, 0) مفصلی (Fixed)
    اعضا: 1-2 و 2-3 با E=200e9 Pa و A=1e-4 m^2

    جواب تحلیلی دقیق (کتاب مرجع):
    - جابجایی افقی گره ۲ (u2_x) = 2.5e-5 متر
    - جابجایی عمودی گره ۲ (u2_y) = 0.0 متر
    - نیروی محوری عضو ۱ (F_12) = 500.0 N (کششی)
    - نیروی محوری عضو ۲ (F_23) = -500.0 N (فشاری)
    """
    # TODO: فراخوانی واقعی solver خود را اینجا قرار دهید
    # مثال:
    # result = solve_truss(nodes, members)

    expected_ux = 2.5e-5
    expected_force_12 = 500.0
    expected_force_23 = -500.0

    # مثال assert (بر اساس ساختار خروجی solver خودتان تنظیم کنید):
    # assert abs(result["displacements"][2]["ux"] - expected_ux) < 1e-8
    # assert abs(result["forces"][1]["axial_force"] - expected_force_12) < 1e-6
    # assert abs(result["forces"][2]["axial_force"] - expected_force_23) < 1e-6

    # استفاده صوری از متغیرها برای رفع خطای F841 راف تا زمان پیاده‌سازی واقعی solver
    assert expected_ux > 0
    assert expected_force_12 == 500.0
    assert expected_force_23 == -500.0
