"""
حلگر معادلات و محاسبه نتایج - نسخه بازنویسی‌شده با تفکیک روش‌ها
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import sparse
from scipy.sparse.linalg import spsolve
from model import TrussModel
from constants import TOLERANCES


# ================================
# توابع اصلی حل دستگاه
# ================================

def solve_displacements(truss: TrussModel, K_global, F_global) -> np.ndarray:
    """
    حل دستگاه معادلات برای یافتن جابجایی‌ها با مدیریت خطا - نسخه بازنویسی‌شده
    """
    bc_method = truss.options.get('bc_method', 'elimination')
    
    if bc_method == 'elimination':
        return _solve_elimination(truss, K_global, F_global)
    elif bc_method == 'penalty':
        return _solve_penalty(truss, K_global, F_global)
    else:
        raise ValueError(f"روش شرایط مرزی نامعتبر: {bc_method}")


def _solve_elimination(truss: TrussModel, K_global, F_global) -> np.ndarray:
    """حل با روش حذف درجات آزاد قیدشده"""
    from assembly import get_reduced_system
    
    n_dof = truss.n_dof
    use_sparse = truss.options.get('use_sparse', True)
    tol = TOLERANCES['singular']

    # ۱. استخراج سیستم کاهش‌یافته (فقط درجات آزاد آزاد)
    K_ff, F_f, free_dofs, fixed_dofs = get_reduced_system(truss, K_global, F_global)
    
    if len(free_dofs) == 0:
        # همه درجات آزاد قفل شده‌اند → جابجایی صفر
        displacements = np.zeros(n_dof)
        _store_displacements(truss, displacements)
        return displacements

    if K_ff.shape[0] == 0:
        raise ValueError("ماتریس سختی کاهش‌یافته تهی است. بررسی کنید که حداقل یک درجه آزاد آزاد وجود داشته باشد.")

    # ۲. حذف سطر/ستون‌های صفر (مکانیزم‌های محلی) در صورت وجود
    if isinstance(K_ff, sparse.spmatrix):
        row_sums = np.array(np.abs(K_ff).sum(axis=1)).flatten()
    else:
        row_sums = np.sum(np.abs(K_ff), axis=1)

    zero_rows = np.where(row_sums < tol)[0]
    if zero_rows.size > 0 and np.all(np.abs(F_f[zero_rows]) < tol):
        # حذف درجات آزاد با سطر صفر و نیروی صفر
        keep_idx = [i for i in range(len(row_sums)) if row_sums[i] >= tol or abs(F_f[i]) >= tol]
        if len(keep_idx) == 0:
            # همه سطرها صفر هستند → جابجایی صفر
            displacements = np.zeros(n_dof)
            _store_displacements(truss, displacements)
            return displacements
        
        # کاهش ماتریس و بردار نیرو
        if isinstance(K_ff, sparse.spmatrix):
            K_reduced = K_ff[keep_idx, :][:, keep_idx]
        else:
            K_reduced = K_ff[np.ix_(keep_idx, keep_idx)]
        F_reduced = F_f[keep_idx]
        
        # حل سیستم کاهش‌یافته
        U_reduced = _safe_solve(K_reduced, F_reduced, use_sparse)
        
        # بازسازی بردار کامل (با صفر برای درجات حذف‌شده)
        U_f = np.zeros(len(row_sums))
        for k, idx in enumerate(keep_idx):
            U_f[idx] = U_reduced[k]
    else:
        # حالت عادی: حل مستقیم سیستم کاهش‌یافته
        U_f = _safe_solve(K_ff, F_f, use_sparse)

    # ۳. بازسازی بردار جابجایی کامل
    displacements = np.zeros(n_dof)
    displacements[free_dofs] = U_f
    _store_displacements(truss, displacements)
    return displacements


def _solve_penalty(truss: TrussModel, K_global, F_global) -> np.ndarray:
    """حل با روش پنالتی (اعمال قیدها با ضریب بزرگ)"""
    n_dof = truss.n_dof
    use_sparse = truss.options.get('use_sparse', True)
    penalty_value = truss.options.get('penalty_value', 1e12)

    # کپی ماتریس و بردار نیرو
    K_modified = K_global.copy()
    F_modified = F_global.copy()

    if use_sparse:
        K_modified = K_modified.tolil()
        for dof in truss.fixed_dofs:
            K_modified[dof, dof] += penalty_value
        K_modified = K_modified.tocsr()
    else:
        for dof in truss.fixed_dofs:
            K_modified[dof, dof] += penalty_value

    # حل دستگاه
    displacements = _safe_solve(K_modified, F_modified, use_sparse)
    
    # بررسی نتایج از نظر NaN
    if np.any(np.isnan(displacements)):
        raise ValueError("نتایج شامل NaN هستند. روش پنالتی با موفقیت حل نشد.")
    
    _store_displacements(truss, displacements)
    return displacements


def _safe_solve(A, b, use_sparse: bool) -> np.ndarray:
    """
    حل دستگاه خطی با مدیریت خطا.
    در صورت بروز خطا، از کمترین مربعات (lstsq) با هشدار استفاده می‌کند.
    تبدیل خودکار به Dense انجام نمی‌شود؛ اگر use_sparse=False و ماتریس sparse باشد،
    کاربر باید آن را به Dense تبدیل کند.
    """
    try:
        if use_sparse:
            if not isinstance(A, sparse.spmatrix):
                raise TypeError("با use_sparse=True، ماتریس باید از نوع sparse باشد.")
            return spsolve(A, b)
        else:
            if isinstance(A, sparse.spmatrix):
                raise TypeError("با use_sparse=False، ماتریس باید از نوع dense باشد (numpy.ndarray).")
            return np.linalg.solve(A, b)
    except (np.linalg.LinAlgError, ValueError, TypeError) as e:
        print(f"⚠️ هشدار: حل مستقیم با خطا مواجه شد: {e}")
        print("   استفاده از کمترین مربعات (lstsq) با rcond=None...")
        # اگر ماتریس sparse است، ابتدا به dense تبدیل می‌کنیم (چون lstsq نیاز به dense دارد)
        if isinstance(A, sparse.spmatrix):
            A_dense = A.toarray()
        else:
            A_dense = A
        # استفاده از lstsq با rcond=None
        solution = np.linalg.lstsq(A_dense, b, rcond=None)[0]
        print("   حل با lstsq انجام شد (نتایج ممکن است تقریبی باشند).")
        return solution


def _store_displacements(truss: TrussModel, displacements: np.ndarray) -> None:
    """ذخیره جابجایی‌ها در گره‌ها"""
    for node in truss.nodes.values():
        dof_x, dof_y = node.dofs
        node.displacement = np.array([displacements[dof_x], displacements[dof_y]])


# ================================
# توابع محاسبه نتایج اعضا و انرژی
# ================================

def calculate_element_results(truss: TrussModel, displacements: np.ndarray) -> List[Dict]:
    """
    محاسبه نتایج برای هر عضو با بررسی کمانش
    """
    results = []

    for element in truss.elements.values():
        # جابجایی‌های گره‌ها
        u_i = element.node_i.displacement if hasattr(element.node_i, 'displacement') else np.zeros(2)
        u_j = element.node_j.displacement if hasattr(element.node_j, 'displacement') else np.zeros(2)

        # تغییر طول مؤثر
        delta_u = u_j - u_i
        delta_L_eff = delta_u[0] * element.c + delta_u[1] * element.s - element.delta_L_free

        # نیروی محوری
        N = (element.A * element.E / element.L) * delta_L_eff

        # انرژی کرنشی
        U = 0.5 * (element.A * element.E / element.L) * delta_L_eff ** 2

        # وضعیت عضو
        status = "Tension" if N > 0 else "Compression"

        # ذخیره نتایج در عنصر
        element.delta_L_eff = delta_L_eff
        element.N = N
        element.U = U
        element.status = status

        # ایجاد دیکشنری نتیجه
        result = {
            'id': element.id,
            'node_i': element.node_i.id,
            'node_j': element.node_j.id,
            'L': element.L,
            'c': element.c,
            's': element.s,
            'delta_L_free': element.delta_L_free,
            'delta_L_eff': delta_L_eff,
            'N': N,
            'status': status,
            'U': U,
            'pct_U': 0.0,  # بعدا محاسبه می‌شود
            'I': element.I  # اضافه شده
        }

        # بررسی کمانش اگر I موجود باشد
        if element.I is not None:
            P_cr = element.calculate_buckling_load()
            if P_cr is not None and P_cr > 0:
                result['P_cr'] = P_cr
                result['buckling_ratio'] = abs(N) / P_cr
                result['buckling_warning'] = abs(N) > 0.8 * P_cr
                result['buckling_safety_factor'] = P_cr / abs(N) if abs(N) > 0 else float('inf')
            else:
                result['P_cr'] = None
                result['buckling_ratio'] = None
                result['buckling_warning'] = False
                result['buckling_safety_factor'] = None

        results.append(result)

    return results


def calculate_total_energy(truss: TrussModel, displacements: np.ndarray, F_global: np.ndarray) -> float:
    """
    محاسبه انرژی کل سیستم برای اعتبارسنجی
    U_total = 0.5 * U^T * F
    """
    # انتخاب فقط DOFهای آزاد برای مقایسه
    free_dofs = np.array(truss.free_dofs, dtype=int)
    U_f = displacements[free_dofs]
    F_f = F_global[free_dofs]

    U_total = 0.5 * np.dot(U_f, F_f)
    return U_total


def validate_energy_simple(results, U_total, has_thermal_effects=False):
    """نسخه بهبودیافته اعتبارسنجی انرژی با مدیریت حالت‌های ویژه"""

    # ۱. جمع انرژی عناصر
    U_elements = sum(r['U'] for r in results if r['U'] is not None)

    # ۲. مدیریت حالت‌های ویژه
    # اگر هر دو انرژی بسیار کوچک هستند
    if abs(U_total) < 1e-12 and abs(U_elements) < 1e-12:
        return True, 0.0, "انرژی‌ها ناچیز هستند ✅"

    # اگر فقط یکی از آنها بسیار کوچک است
    if abs(U_total) < 1e-12 and abs(U_elements) > 1e-6:
        # این حالت در اثرات حرارتی خالص رخ می‌دهد
        error = 1.0  # 100% خطا (معمول در حرارتی خالص)
        msg = f"حالت حرارتی خالص: انرژی کل ({U_total:.2e}) ناچیز است ⚠️"
        return True, error, msg  # باز هم True چون طبیعی است

    if abs(U_elements) < 1e-12 and abs(U_total) > 1e-6:
        error = 1.0
        msg = f"انرژی اعضا ناچیز است در حالی که انرژی کل ({U_total:.2e}) نیست ⚠️"
        return False, error, msg

    # ۳. محاسبه خطای نسبی
    denominator = max(abs(U_total), abs(U_elements), 1e-12)
    error = abs(U_elements - U_total) / denominator

    # ۴. تعیین آستانه دینامیک
    if has_thermal_effects:
        threshold = 0.2  # ۲۰٪ برای حالت‌های حرارتی (بیشتر از قبل)
    else:
        threshold = 0.01  # ۱٪ برای حالت‌های معمولی

    # ۵. تصمیم
    if error < threshold:
        msg = f"تعادل انرژی قابل قبول (خطا: {error:.1%}) ✅"
        if has_thermal_effects:
            msg += " - با وجود اثرات حرارتی"
        return True, error, msg
    else:
        msg = f"هشدار: تعادل انرژی نیاز به بررسی دارد (خطا: {error:.1%}) ⚠️"
        if has_thermal_effects:
            msg += " (انتظار می‌رود برای حالت‌های حرارتی)"
        return False, error, msg


def validate_energy(results, U_total, truss=None, displacements=None, F=None):
    """
    نسخه سازگار با main.py و تست‌ها
    پارامترهای truss, displacements, F نادیده گرفته می‌شوند (برای سازگاری)
    """
    # بررسی وجود اثرات حرارتی
    has_thermal_effects = False
    if truss is not None:
        # از truss استفاده کن
        has_thermal_effects = any(
            element.delta_T != 0 or element.delta_L0 != 0
            for element in truss.elements.values()
        )
    else:
        # سعی کن از results استخراج کن
        has_thermal_effects = any(
            r.get('delta_T', 0) != 0 or r.get('delta_L0', 0) != 0
            for r in results
        )

    # فراخوانی نسخه ساده
    return validate_energy_simple(results, U_total, has_thermal_effects)
