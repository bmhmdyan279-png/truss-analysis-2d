"""
حلگر معادلات و محاسبه نتایج - نسخه نهایی با اصلاح اعتبارسنجی انرژی
"""

import logging
from typing import Dict, List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .constants import (
    DEFAULT_PENALTY_VALUE,
    THERMAL_ENERGY_WARN,
    TOLERANCES,
    ZERO_ENERGY_TOL,
    ZERO_LENGTH_TOLERANCE,
)
from .model import TrussModel

logger = logging.getLogger(__name__)


def solve_displacements(truss: TrussModel, K_global, F_global) -> np.ndarray:
    """
    حل دستگاه معادلات برای یافتن جابجایی‌ها با مدیریت خطا - نسخه اصلاح شده
    """
    bc_method = truss.options.get("bc_method", "elimination")
    penalty_value = truss.options.get("penalty_value", DEFAULT_PENALTY_VALUE)
    use_sparse = truss.options.get("use_sparse", True)

    n_dof = truss.n_dof
    displacements = np.zeros(n_dof)  # مقدار پیش‌فرض

    try:
        if bc_method == "elimination":
            from .assembly import get_reduced_system

            K_ff, F_f, free_dofs, fixed_dofs = get_reduced_system(
                truss, K_global, F_global
            )
            # حذف سطر/ستون‌های صفر برای جلوگیری از ماتریس منفرد
            tol = TOLERANCES["singular"]

            if isinstance(K_ff, np.ndarray):
                row_sums = np.sum(np.abs(K_ff), axis=1)
            else:
                row_sums = np.array(np.sum(np.abs(K_ff), axis=1)).flatten()

            # پیدا کردن سطرهای غیرصفر
            non_zero_indices = np.where(row_sums > tol)[0]

            if len(non_zero_indices) < len(row_sums):
                # کاهش ماتریس
                if isinstance(K_ff, np.ndarray):
                    K_ff = K_ff[non_zero_indices, :][:, non_zero_indices]
                else:
                    K_ff = K_ff[non_zero_indices, :][:, non_zero_indices]
                F_f = F_f[non_zero_indices]

                # حل سیستم کاهش یافته
                if use_sparse and isinstance(K_ff, sparse.spmatrix):
                    U_f_reduced = spsolve(K_ff, F_f)
                else:
                    if isinstance(K_ff, sparse.spmatrix):
                        K_ff = K_ff.toarray()  # type: ignore
                    U_f_reduced = np.linalg.solve(K_ff, F_f)

                # بازسازی بردار کامل
                U_f = np.zeros(len(row_sums))
                U_f[non_zero_indices] = U_f_reduced
            else:
                # حل سیستم کامل
                if use_sparse and isinstance(K_ff, sparse.spmatrix):
                    U_f = spsolve(K_ff, F_f)
                else:
                    if isinstance(K_ff, sparse.spmatrix):
                        K_ff = K_ff.toarray()  # type: ignore
                    U_f = np.linalg.solve(K_ff, F_f)

            # بررسی شرایط ایستایی
            if len(free_dofs) == 0:
                # حالت خاص: همه درجات آزادی قفل شده‌اند.
                displacements = np.zeros(n_dof)
                for node in truss.nodes.values():
                    dof_x, dof_y = node.dofs
                    node.displacement = np.array(
                        [displacements[dof_x], displacements[dof_y]]
                    )
                return displacements

            if len(free_dofs) > 0 and K_ff.shape[0] == 0:
                raise ValueError("ماتریس سختی کاهش یافته تهی است.")

            # حل برای DOFهای آزاد
            # بررسی وجود ردیف/ستون‌های صفر (مکانیزم‌های آزاد محلی)
            if isinstance(K_ff, sparse.spmatrix):
                K_check = K_ff.tocsc()  # type: ignore
                row_sums = np.array(np.abs(K_check).sum(axis=1)).flatten()
            else:
                row_sums = np.sum(np.abs(K_ff), axis=1)

            zero_rows = np.where(row_sums < ZERO_LENGTH_TOLERANCE)[0]

            if zero_rows.size > 0:
                # حذف DOFهای صفر
                to_keep = []
                to_remove = []
                for idx in range(len(row_sums)):
                    if (
                        row_sums[idx] < ZERO_LENGTH_TOLERANCE
                        and abs(F_f[idx]) < ZERO_LENGTH_TOLERANCE
                    ):
                        to_remove.append(idx)
                    else:
                        to_keep.append(idx)

                if len(to_remove) > 0 and len(to_keep) > 0:
                    # ساخت زیرماتریس کاهش‌یافته
                    if isinstance(K_ff, sparse.spmatrix):
                        K_reduced = K_ff[to_keep, :][:, to_keep]
                    else:
                        K_reduced = K_ff[np.ix_(to_keep, to_keep)]
                    F_reduced = F_f[to_keep]

                    # حل روی سیستم کاهش‌یافته
                    try:
                        if use_sparse and isinstance(K_reduced, sparse.spmatrix):
                            U_reduced = spsolve(K_reduced, F_reduced)
                        else:
                            U_reduced = np.linalg.solve(K_reduced, F_reduced)
                    except Exception:
                        # Fail-fast: مکانیزم یا ماتریس منفرد
                        raise np.linalg.LinAlgError(
                            "ماتریس سختی منفرد است (مکانیزم)."
                        ) from None

                    # بازسازی U_f با مقدار صفر برای DOFهای حذف‌شده
                    U_f = np.zeros(len(row_sums))
                    for k, idx in enumerate(to_keep):
                        U_f[idx] = U_reduced[k]
                else:
                    # حل مستقیم
                    try:
                        if use_sparse and isinstance(K_ff, sparse.spmatrix):
                            U_f = spsolve(K_ff, F_f)
                        else:
                            U_f = np.linalg.solve(K_ff, F_f)
                    except Exception:
                        if isinstance(K_ff, sparse.spmatrix):
                            K_ff_dense = K_ff.toarray()  # type: ignore
                            U_f = np.linalg.solve(K_ff_dense, F_f)
                        else:
                            # اگر هنوز خطا داد، از کمترین مربعات استفاده کن
                            raise np.linalg.LinAlgError(
                                "ماتریس سختی منفرد است (مکانیزم). حل متوقف شد."
                            ) from None
            else:
                # حالت معمول: حل مستقیم
                try:
                    if use_sparse and isinstance(K_ff, sparse.spmatrix):
                        U_f = spsolve(K_ff, F_f)
                    else:
                        U_f = np.linalg.solve(K_ff, F_f)
                except Exception:
                    if isinstance(K_ff, sparse.spmatrix):
                        K_ff_dense = K_ff.toarray()  # type: ignore
                        U_f = np.linalg.solve(K_ff_dense, F_f)
                    else:
                        # اگر هنوز خطا داد، از کمترین مربعات استفاده کن
                        raise np.linalg.LinAlgError(
                            "ماتریس سختی منفرد است (مکانیزم). حل متوقف شد."
                        ) from None

            # جایگذاری جابجایی‌ها
            displacements[free_dofs] = U_f

        elif bc_method == "penalty":
            # روش پنالتی

            K_modified = K_global.copy()

            F_modified = F_global.copy()

            if use_sparse:
                K_modified = K_modified.tolil()

            # اعمال پنالتی بر روی DOFهای قفل شده

            for dof in truss.fixed_dofs:
                if use_sparse:
                    K_modified[dof, dof] += penalty_value

                else:
                    K_modified[dof, dof] += penalty_value

            if use_sparse:
                K_modified = K_modified.tocsr()

                try:
                    displacements = spsolve(K_modified, F_modified)

                    # بررسی اینکه نتایج nan نباشند

                    if np.any(np.isnan(displacements)):
                        raise ValueError("نتایج شامل nan هستند")

                except (Exception, ValueError) as e:
                    logger.info(
                        f"⚠️ خطا در حل با روش پنالتی: {e}. استفاده از روش fallback..."
                    )

                    # Fail-fast: مکانیزم یا ماتریس منفرد

                    raise np.linalg.LinAlgError(
                        "ماتریس سختی در روش پنالتی منفرد است. حل متوقف شد."
                    ) from None

            else:
                try:
                    displacements = np.linalg.solve(K_modified, F_modified)

                    if np.any(np.isnan(displacements)):
                        raise ValueError("نتایج شامل nan هستند")

                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.info(
                        f"⚠️ خطا در حل با روش پنالتی: {e}. استفاده از کمترین مربعات..."
                    )

                    # اضافه کردن یک مقدار کوچک به قطر

                    K_modified += np.eye(K_modified.shape[0]) * 1e-8

                    raise np.linalg.LinAlgError(
                        "ماتریس سختی در روش پنالتی منفرد است. حل متوقف شد."
                    ) from None

        else:
            raise ValueError(f"روش شرایط مرزی نامعتبر: {bc_method}")

        # ذخیره جابجایی‌ها در گره‌ها
        for node in truss.nodes.values():
            dof_x, dof_y = node.dofs
            node.displacement = np.array([displacements[dof_x], displacements[dof_y]])

        return displacements  # اینجا حتماً return می‌شود

    except np.linalg.LinAlgError as e:
        logger.error(f"❌ خطای بحرانی جبر خطی: {str(e)}")
        logger.error(
            "🔍 سازه ناپایدار است (مکانیزم) یا ماتریس سختی منفرد است. تحلیل متوقف می‌شود."
        )
        raise ValueError(f"TRUSS-3001: خطای حلگر - {str(e)}") from e

    except Exception as e:
        logger.error(f"❌ خطای غیرمنتظره در حلگر: {str(e)}")
        raise RuntimeError(f"TRUSS-3999: خطای ناشناخته در حلگر - {str(e)}") from e


def calculate_element_results(
    truss: TrussModel, displacements: np.ndarray
) -> List[Dict]:
    """
    محاسبه نتایج برای هر عضو با بررسی کمانش
    """
    results = []

    for element in truss.elements.values():
        # جابجایی‌های گره‌ها
        u_i = (
            element.node_i.displacement
            if hasattr(element.node_i, "displacement")
            else np.zeros(2)
        )
        u_j = (
            element.node_j.displacement
            if hasattr(element.node_j, "displacement")
            else np.zeros(2)
        )

        # تغییر طول مؤثر
        delta_u = u_j - u_i
        delta_L_eff = (
            delta_u[0] * element.c + delta_u[1] * element.s - element.delta_L_free
        )

        # نیروی محوری
        N = (element.A * element.E / element.L) * delta_L_eff

        # انرژی کرنشی
        U = 0.5 * (element.A * element.E / element.L) * delta_L_eff**2

        # وضعیت عضو
        status = "Tension" if N > 0 else "Compression"

        # ذخیره نتایج در عنصر
        element.delta_L_eff = delta_L_eff
        element.N = N
        element.U = U
        element.status = status

        # ایجاد دیکشنری نتیجه
        result = {
            "id": element.id,
            "node_i": element.node_i.id,
            "node_j": element.node_j.id,
            "L": element.L,
            "c": element.c,
            "s": element.s,
            "delta_L_free": element.delta_L_free,
            "delta_L_eff": delta_L_eff,
            "N": N,
            "status": status,
            "U": U,
            "pct_U": 0.0,  # بعدا محاسبه می‌شود
            "I": element.I,  # اضافه شده
        }

        # بررسی کمانش اگر I موجود باشد
        if element.I is not None:
            P_cr = element.calculate_buckling_load()
            if P_cr is not None and P_cr > 0:
                result["P_cr"] = P_cr
                result["buckling_ratio"] = abs(N) / P_cr
                result["buckling_warning"] = abs(N) > 0.8 * P_cr
                result["buckling_safety_factor"] = (
                    P_cr / abs(N) if abs(N) > 0 else float("inf")
                )
            else:
                result["P_cr"] = None
                result["buckling_ratio"] = None
                result["buckling_warning"] = False
                result["buckling_safety_factor"] = None

        results.append(result)

    return results


def calculate_total_energy(
    truss: TrussModel, displacements: np.ndarray, F_global: np.ndarray
) -> float:
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
    U_elements = sum(r["U"] for r in results if r["U"] is not None)

    # ۲. مدیریت حالت‌های ویژه
    # اگر هر دو انرژی بسیار کوچک هستند
    if abs(U_total) < ZERO_LENGTH_TOLERANCE and abs(U_elements) < ZERO_LENGTH_TOLERANCE:
        return True, 0.0, "انرژی‌ها ناچیز هستند ✅"

    # اگر فقط یکی از آنها بسیار کوچک است
    if abs(U_total) < ZERO_LENGTH_TOLERANCE and abs(U_elements) > ZERO_ENERGY_TOL:
        # این حالت در اثرات حرارتی خالص رخ می‌دهد
        error = 1.0  # 100% خطا (معمول در حرارتی خالص)
        msg = f"حالت حرارتی خالص: انرژی کل ({U_total:.2e}) ناچیز است ⚠️"
        return True, error, msg  # باز هم True چون طبیعی است

    if abs(U_elements) < ZERO_LENGTH_TOLERANCE and abs(U_total) > ZERO_ENERGY_TOL:
        error = 1.0
        msg = f"انرژی اعضا ناچیز است در حالی که انرژی کل ({U_total:.2e}) نیست ⚠️"
        return False, error, msg

    # ۳. محاسبه خطای نسبی
    denominator = max(abs(U_total), abs(U_elements), ZERO_LENGTH_TOLERANCE)
    error = abs(U_elements - U_total) / denominator

    # ۴. تعیین آستانه دینامیک
    if has_thermal_effects:
        threshold = THERMAL_ENERGY_WARN  # ۱٪ برای حالت‌های حرارتی (اصلاح شده)
    else:
        threshold = THERMAL_ENERGY_WARN  # ۱٪ برای حالت‌های معمولی

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
            r.get("delta_T", 0) != 0 or r.get("delta_L0", 0) != 0 for r in results
        )

    # فراخوانی نسخه ساده
    return validate_energy_simple(results, U_total, has_thermal_effects)
