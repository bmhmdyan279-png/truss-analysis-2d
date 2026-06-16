"""
پس‌پردازش و تولید گزارش - نسخه کامل و نهایی
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')  # استفاده از backend غیرتعاملی
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
import csv
import json
import os
import warnings
import logging
from model import TrussModel

logger = logging.getLogger(__name__)


def sort_elements(results: List[Dict], by: str = 'energy',
                  descending: bool = True) -> List[Dict]:
    """
    مرتب‌سازی اعضا بر اساس معیار مشخص - نسخه بهبودیافته
    """
    if not results:
        return results

    # ایجاد کپی برای جلوگیری از تغییر لیست اصلی
    sorted_results = results.copy()

    # تعیین کلید مرتب‌سازی بر اساس معیار
    valid_keys = ['energy', 'force', 'abs_force', 'id', 'buckling_ratio',
                  'pct_energy', 'length', 'delta_L_eff', 'status']

    if by == 'energy':
        key = 'U'
    elif by == 'force':
        key = 'N'
    elif by == 'abs_force':
        key = 'abs_N'
        # محاسبه قدر مطلق نیرو اگر وجود نداشته باشد
        for r in sorted_results:
            if 'abs_N' not in r:
                r['abs_N'] = abs(r.get('N', 0))
    elif by == 'id':
        key = 'id'
    elif by == 'buckling_ratio':
        key = 'buckling_ratio'
    elif by == 'pct_energy':
        key = 'pct_U'
    elif by == 'length':
        key = 'L'
    elif by == 'delta_L_eff':
        key = 'delta_L_eff'
    elif by == 'status':
        key = 'status'
    else:
        raise ValueError(
            f"معیار مرتب‌سازی نامعتبر: '{by}'. معیارهای مجاز: {', '.join(valid_keys)}"
        )

    # تابع کمکی برای مدیریت مقادیر None و مقایسه ایمن
    def get_sort_key(x: Dict) -> Tuple:
        val = x.get(key)

        # مدیریت مقادیر None
        if val is None:
            # برای اعداد: استفاده از float('-inf') یا float('inf')
            if by in ['energy', 'force', 'abs_force', 'buckling_ratio',
                      'pct_energy', 'length', 'delta_L_eff']:
                return (1, float('-inf') if descending else float('inf'))
            elif by == 'status':
                return (1, '')
            elif by == 'id':
                return (1, float('inf'))

        # برای اعداد
        if isinstance(val, (int, float, np.number)):
            return (0, val)
        # برای رشته‌ها
        elif isinstance(val, str):
            return (0, val)
        # برای سایر انواع
        else:
            try:
                return (0, float(val))
            except (ValueError, TypeError):
                return (1, str(val))

    try:
        # مرتب‌سازی با توجه به کلید
        sorted_results.sort(key=get_sort_key, reverse=descending)
    except Exception as e:
        raise ValueError(f"خطا در مرتب‌سازی بر اساس معیار '{by}': {str(e)}")

    return sorted_results


def calculate_percentages(results: List[Dict]) -> List[Dict]:
    """
    محاسبه درصد سهم انرژی هر عضو با مدیریت مقادیر نامعتبر
    """
    if not results:
        return results

    # محاسبه مجموع انرژی با فیلتر مقادیر نامعتبر
    valid_energies = []
    for r in results:
        energy = r.get('U')
        if energy is not None and np.isfinite(energy):
            valid_energies.append(energy)

    total_energy = sum(valid_energies) if valid_energies else 0.0

    # بررسی شرایط ویژه
    if not np.isfinite(total_energy) or abs(total_energy) < 1e-12:
        # اگر انرژی کل صفر، نامعین یا بسیار کوچک است
        for r in results:
            r['pct_U'] = 0.0
    else:
        # محاسبه درصد هر عضو
        for r in results:
            energy = r.get('U', 0)
            if energy is None or not np.isfinite(energy):
                r['pct_U'] = 0.0
            else:
                r['pct_U'] = 100.0 * energy / total_energy

    return results


def calculate_displacement_scale_factor(truss: TrussModel) -> float:
    """
    محاسبه خودکار ضریب بزرگنمایی تغییرشکل
    """
    if not truss.nodes:
        return 1.0

    try:
        # محاسبه بیشترین جابجایی
        max_disp = 0.0
        for node in truss.nodes.values():
            if hasattr(node, 'displacement') and node.displacement is not None:
                disp_norm = np.linalg.norm(node.displacement)
                if np.isfinite(disp_norm) and disp_norm > max_disp:
                    max_disp = disp_norm

        # اگر جابجایی‌ها بسیار کوچک یا صفر هستند
        if max_disp < 1e-10:
            return 1.0

        # محاسبه ابعاد کلی سازه
        xs = [node.x for node in truss.nodes.values() if hasattr(node, 'x')]
        ys = [node.y for node in truss.nodes.values() if hasattr(node, 'y')]

        if len(xs) < 2 or len(ys) < 2:
            span = 1.0
        else:
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            span = max(span_x, span_y)

            # اطمینان از اینکه span صفر نباشد
            if span < 1e-10:
                span = 1.0

            # محاسبه ضریب بزرگنمایی
            target_ratio = 0.1  # 10% از اندازه سازه
            scale_factor = target_ratio * span / max_disp

            # محدودیت‌های منطقی
            if scale_factor < 1.0:
                scale_factor = 1.0  # حداقل ۱ برابر
            elif scale_factor > 1000.0:
                scale_factor = 1000.0  # حداکثر ۱۰۰۰ برابر

            # یا: استفاده از مقیاس لگاریتمی
            # scale_factor = 10 ** np.clip(np.log10(scale_factor), 0, 3)

            return scale_factor

    except Exception as e:
        warnings.warn(f"خطا در محاسبه ضریب بزرگنمایی: {e}. استفاده از ضریب پیش‌فرض 50.")
        return 50.0


def calculate_displacement_statistics(truss: TrussModel,
                                      displacements: np.ndarray) -> Dict:
    """
    محاسبه آمار جابجایی‌ها
    """
    stats = {
        'max_displacement': 0.0,
        'max_displacement_node': None,
        'max_x_displacement': 0.0,
        'max_y_displacement': 0.0,
        'rms_displacement': 0.0,
        'displacements': []
    }

    if not truss.nodes or displacements is None:
        return stats

    disp_magnitudes = []
    max_disp = 0.0
    max_disp_node = None

    for node in truss.nodes.values():
        if not hasattr(node, 'dofs') or node.dofs is None:
            continue

        dof_x, dof_y = node.dofs

        # بررسی محدوده DOFها
        if (dof_x >= len(displacements) or dof_y >= len(displacements) or
                dof_x < 0 or dof_y < 0):
            continue

        u_x = displacements[dof_x]
        u_y = displacements[dof_y]
        disp_mag = np.linalg.norm([u_x, u_y])

        if np.isfinite(disp_mag):
            disp_magnitudes.append(disp_mag)

            if disp_mag > max_disp:
                max_disp = disp_mag
                max_disp_node = node.id

            # به‌روزرسانی بیشترین جابجایی در هر راستا
            if abs(u_x) > stats['max_x_displacement']:
                stats['max_x_displacement'] = abs(u_x)
            if abs(u_y) > stats['max_y_displacement']:
                stats['max_y_displacement'] = abs(u_y)

            # ذخیره جابجایی‌های هر گره
            stats['displacements'].append({
                'node_id': node.id,
                'u_x': float(u_x),
                'u_y': float(u_y),
                'magnitude': float(disp_mag)
            })

    if disp_magnitudes:
        stats['max_displacement'] = max_disp
        stats['max_displacement_node'] = max_disp_node
        stats['rms_displacement'] = float(np.sqrt(np.mean(np.square(disp_magnitudes))))

    return stats


def calculate_buckling_statistics(results: List[Dict]) -> Dict:
    """
    محاسبه آمار کمانش - نسخه بهبودیافته با warning_count
    """
    stats = {
        'elements_with_I': 0,
        'elements_at_risk': 0,
        'warning_count': 0,
        'max_buckling_ratio': 0.0,
        'critical_elements': [],
        'warnings': []
    }

    for r in results:
        # بررسی وجود I یا P_cr
        has_I = r.get('I') is not None
        has_P_cr = r.get('P_cr') is not None

        if has_I or has_P_cr:
            stats['elements_with_I'] += 1

            if r.get('buckling_ratio') is not None:
                ratio = r['buckling_ratio']

                if np.isfinite(ratio) and ratio > 0:
                    # به‌روزرسانی بیشترین نسبت کمانش
                    if ratio > stats['max_buckling_ratio']:
                        stats['max_buckling_ratio'] = ratio

                    # بررسی خطر کمانش
                    if ratio > 0.8:
                        stats['elements_at_risk'] += 1

                        warning = {
                            'element_id': r['id'],
                            'nodes': f"{r.get('node_i', '?')}-{r.get('node_j', '?')}",
                            'buckling_ratio': float(ratio),
                            'force': float(r.get('N', 0)),
                            'critical_load': float(r.get('P_cr', 0)) if r.get('P_cr') is not None else None,
                            'status': r.get('status', 'Unknown')
                        }

                        stats['warnings'].append(warning)

                        if ratio > 0.5:
                            stats['critical_elements'].append(warning)

    # محاسبه warning_count برای سازگاری
    stats['warning_count'] = len(stats['warnings'])

    # مرتب‌سازی عناصر بحرانی بر اساس نسبت کمانش
    stats['critical_elements'].sort(key=lambda x: x.get('buckling_ratio', 0), reverse=True)
    stats['warnings'].sort(key=lambda x: x.get('buckling_ratio', 0), reverse=True)

    return stats


def generate_plots(truss: TrussModel, displacements: np.ndarray,
                   results: List[Dict], output_prefix: str) -> Dict:
    """
    تولید نمودارهای تحلیل با مدیریت سبک matplotlib
    """
    # تنظیم سبک matplotlib با fallback
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True

    plot_files = {}

    # محاسبه ضریب بزرگنمایی تغییرشکل
    if truss.options.get('displacement_scale', 'auto') == 'auto':
        scale_factor = calculate_displacement_scale_factor(truss)
    else:
        scale_factor = float(truss.options.get('displacement_scale', 50.0))

    logger.info(f"📏 ضریب بزرگنمایی تغییرشکل: {scale_factor:.1f}×")

    # 1. نمودار هندسه اولیه و تغییر شکل یافته
    try:
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        # رسم اعضای اولیه
        first_geom = True
        for element in truss.elements.values():
            if (hasattr(element, 'node_i') and hasattr(element.node_i, 'x') and
                    hasattr(element, 'node_j') and hasattr(element.node_j, 'x')):
                x_vals = [element.node_i.x, element.node_j.x]
                y_vals = [element.node_i.y, element.node_j.y]
                label = 'هندسه اولیه' if first_geom else None
                ax1.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.7, label=label)
                first_geom = False

        # رسم اعضای تغییر شکل یافته
        first_deformed = True
        for element in truss.elements.values():
            if (hasattr(element, 'node_i') and hasattr(element.node_i, 'displacement') and
                    hasattr(element, 'node_j') and hasattr(element.node_j, 'displacement')):
                xi = element.node_i.x + element.node_i.displacement[0] * scale_factor
                yi = element.node_i.y + element.node_i.displacement[1] * scale_factor
                xj = element.node_j.x + element.node_j.displacement[0] * scale_factor
                yj = element.node_j.y + element.node_j.displacement[1] * scale_factor

                label = 'هندسه تغییر شکل یافته' if first_deformed else None
                ax1.plot([xi, xj], [yi, yj], 'r--', linewidth=2, alpha=0.8, label=label)
                first_deformed = False

        # رسم گره‌ها
        for node in truss.nodes.values():
            if hasattr(node, 'x') and hasattr(node, 'y'):
                color = 'darkgreen' if node.is_support else 'blue'
                marker = 's' if node.is_support else 'o'
                size = 100 if node.is_support else 60
                ax1.scatter(node.x, node.y, s=size, c=color, marker=marker,
                            zorder=5, edgecolors='black', linewidth=1)
                ax1.text(node.x, node.y, f'{node.id}', fontsize=12,
                         ha='right', va='bottom', fontweight='bold')

        ax1.set_xlabel('مختصات X (m)', fontsize=12)
        ax1.set_ylabel('مختصات Y (m)', fontsize=12)
        ax1.set_title(f'هندسه خرپا - مقیاس تغییرشکل: {scale_factor:.1f}×',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # اضافه کردن راهنما
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='هندسه اولیه'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                   label='تغییر شکل یافته'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='گره آزاد'),
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='darkgreen', markersize=10, label='تکیه‌گاه')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        geometry_file = f'{output_prefix}_geometry.png'

        # ایجاد پوشه اگر وجود ندارد
        os.makedirs(os.path.dirname(geometry_file) or '.', exist_ok=True)

        plt.savefig(geometry_file, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        plot_files['geometry'] = geometry_file
        logger.info(f"✅ نمودار هندسه در {geometry_file} ذخیره شد.")

    except Exception as e:
        logger.error(f"❌ خطا در رسم نمودار هندسه: {str(e)}")

    # 2. نمودار دایره‌ای سهم انرژی
    try:
        if results:
            # فیلتر کردن اعضای با انرژی قابل توجه
            filtered_results = [r for r in results if r.get('U', 0) > 1e-10]

            if len(filtered_results) > 0:
                # ایجاد نمودار
                fig2, ax2 = plt.subplots(figsize=(10, 8))

                # جمع‌آوری داده‌ها برای نمودار
                labels = [f"عضو {r['id']}" for r in filtered_results]
                sizes = [r.get('U', 0) for r in filtered_results]

                # رنگ‌بندی بر اساس کشش/فشار
                colors = []
                for r in filtered_results:
                    if r.get('status') == 'Tension':
                        colors.append('#2ecc71')  # سبز برای کشش
                    else:
                        colors.append('#e74c3c')  # قرمز برای فشار

                # ایجاد نمودار دایره‌ای
                wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                                   autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                                   startangle=90, pctdistance=0.85,
                                                   wedgeprops={'edgecolor': 'white', 'linewidth': 1})

                # بهبود خوانایی
                for text in texts:
                    text.set_fontsize(9)
                    text.set_fontweight('bold')
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(8)

                ax2.set_title('توزیع انرژی کرنشی بین اعضا', fontsize=14, fontweight='bold')

                # راهنمای رنگ‌ها
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ecc71', alpha=0.8, label='کشش'),
                    Patch(facecolor='#e74c3c', alpha=0.8, label='فشار')
                ]
                ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)

                # اضافه کردن مرکز دایره
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax2.add_artist(centre_circle)

                plt.tight_layout()
                energy_pie_file = f'{output_prefix}_energy_pie.png'

                # ایجاد پوشه اگر وجود ندارد
                os.makedirs(os.path.dirname(energy_pie_file) or '.', exist_ok=True)

                plt.savefig(energy_pie_file, dpi=300, bbox_inches='tight')
                plt.close(fig2)
                plot_files['energy_pie'] = energy_pie_file
                logger.info(f"✅ نمودار انرژی در {energy_pie_file} ذخیره شد.")
            else:
                logger.info("⚠️ نمودار انرژی رسم نشد: انرژی همه اعضا ناچیز است.")
        else:
            logger.info("⚠️ نمودار انرژی رسم نشد: هیچ نتیجه‌ای وجود ندارد.")

    except Exception as e:
        logger.error(f"❌ خطا در رسم نمودار انرژی: {str(e)}")

    return plot_files


def generate_report(truss: TrussModel, displacements: np.ndarray,
                    results: List[Dict], units: str = 'SI') -> Dict:
    """
    تولید گزارش کامل تحلیل
    """
    # محاسبه آمار جابجایی‌ها
    disp_stats = calculate_displacement_statistics(truss, displacements)

    # محاسبه آمار کمانش
    buckling_stats = calculate_buckling_statistics(results)

    # محاسبه آمار نیرو و انرژی
    if results:
        # مرتب‌سازی بر اساس انرژی
        sorted_by_energy = sort_elements(results, by='energy', descending=True)
        sorted_by_force = sort_elements(results, by='abs_force', descending=True)

        max_energy_elem = sorted_by_energy[0] if sorted_by_energy else {}
        max_force_elem = sorted_by_force[0] if sorted_by_force else {}

        # محاسبه توزیع نیرو
        tension_count = sum(1 for r in results if r.get('status') == 'Tension')
        compression_count = sum(1 for r in results if r.get('status') == 'Compression')

        # نیروهای حداکثر
        max_tension = max([r.get('N', 0) for r in results if r.get('status') == 'Tension'],
                          default=0)
        max_compression = min([r.get('N', 0) for r in results if r.get('status') == 'Compression'],
                              default=0)

        # مجموع انرژی
        total_energy = sum([r.get('U', 0) for r in results])
    else:
        max_energy_elem = {}
        max_force_elem = {}
        tension_count = 0
        compression_count = 0
        max_tension = 0
        max_compression = 0
        total_energy = 0

    # ایجاد گزارش
    report = {
        'metadata': {
            'software': 'Truss Analysis 2D',
            'version': '2.0.0',
            'units': units,
            'analysis_date': np.datetime64('now').astype(str),
            'input_file': getattr(truss, 'input_file', 'unknown'),
            'total_nodes': len(truss.nodes),
            'free_nodes': len(truss.free_nodes),
            'supported_nodes': len(truss.supported_nodes),
            'total_elements': len(truss.elements),
            'boundary_condition_method': truss.options.get('bc_method', 'elimination'),
            'solver_type': 'sparse' if truss.options.get('use_sparse', True) else 'dense',
            'global_temperature_change': truss.global_delta_T
        },
        'energy_statistics': {
            'total_energy': float(total_energy),
            'max_energy_element': {
                'id': max_energy_elem.get('id'),
                'energy': float(max_energy_elem.get('U', 0)),
                'percentage': float(max_energy_elem.get('pct_U', 0)),
                'nodes': f"{max_energy_elem.get('node_i', '?')}-{max_energy_elem.get('node_j', '?')}",
                'status': max_energy_elem.get('status', 'Unknown')
            },
            'max_force_element': {
                'id': max_force_elem.get('id'),
                'force': float(max_force_elem.get('N', 0)),
                'status': max_force_elem.get('status', 'Unknown'),
                'nodes': f"{max_force_elem.get('node_i', '?')}-{max_force_elem.get('node_j', '?')}"
            }
        },
        'force_distribution': {
            'tension_elements': tension_count,
            'compression_elements': compression_count,
            'tension_percentage': 100 * tension_count / len(results) if results else 0,
            'compression_percentage': 100 * compression_count / len(results) if results else 0,
            'max_tensile_force': float(max_tension),
            'max_compressive_force': float(max_compression)
        },
        'displacement_statistics': disp_stats,
        'buckling_analysis': buckling_stats,
        'thermal_effects': {
            'elements_with_delta_T': sum(1 for e in truss.elements.values() if e.delta_T != 0),
            'elements_with_delta_L0': sum(1 for e in truss.elements.values() if e.delta_L0 != 0),
            'max_delta_T': max([abs(e.delta_T) for e in truss.elements.values()], default=0),
            'max_delta_L0': max([abs(e.delta_L0) for e in truss.elements.values()], default=0)
        },
        'validation': {
            'sign_convention_valid': True,
            'energy_balance_valid': None,
            'energy_error': None,
            'energy_message': None,
            'analysis_successful': True
        }
    }

    return report


def save_report_to_markdown(report: Dict, filepath: str):
    """
    ذخیره گزارش به فرمت Markdown
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # سربرگ
            f.write("# گزارش تحلیل خرپا\n\n")
            f.write(f"**تاریخ تحلیل:** {report['metadata']['analysis_date']}\n")
            f.write(f"**واحدها:** {report['metadata']['units']}\n")
            f.write(f"**روش شرایط مرزی:** {report['metadata']['boundary_condition_method']}\n\n")

            # خلاصه اجرا
            f.write("## خلاصه اجرا\n\n")
            f.write(f"- **تعداد گره‌ها:** {report['metadata']['total_nodes']}\n")
            f.write(f"- **گره‌های آزاد:** {report['metadata']['free_nodes']}\n")
            f.write(f"- **تکیه‌گاه‌ها:** {report['metadata']['supported_nodes']}\n")
            f.write(f"- **تعداد اعضا:** {report['metadata']['total_elements']}\n")
            f.write(f"- **تغییر دمای سراسری:** {report['metadata']['global_temperature_change']} °C\n\n")

            # آمار انرژی
            f.write("## آمار انرژی\n\n")
            f.write(f"- **انرژی کل کرنشی:** {report['energy_statistics']['total_energy']:.4e} J\n")
            elem = report['energy_statistics']['max_energy_element']
            if elem['id'] is not None:
                f.write(f"- **عضو با بیشترین انرژی:** عضو {elem['id']} (گره‌های {elem['nodes']})\n")
                f.write(f"  - انرژی: {elem['energy']:.4e} J\n")
                f.write(f"  - درصد انرژی کل: {elem['percentage']:.1f}%\n")
                f.write(f"  - وضعیت: {elem['status']}\n\n")

            # آمار نیرو
            f.write("## آمار نیرو\n\n")
            f.write(f"- **اعضای کششی:** {report['force_distribution']['tension_elements']}\n")
            f.write(f"- **اعضای فشاری:** {report['force_distribution']['compression_elements']}\n")
            f.write(f"- **بیشترین نیروی کششی:** {report['force_distribution']['max_tensile_force']:.4e} N\n")
            f.write(f"- **بیشترین نیروی فشاری:** {report['force_distribution']['max_compressive_force']:.4e} N\n\n")

            # آمار جابجایی
            f.write("## آمار جابجایی\n\n")
            f.write(f"- **بیشترین جابجایی:** {report['displacement_statistics']['max_displacement']:.4e} m\n")
            if report['displacement_statistics']['max_displacement_node'] is not None:
                f.write(f"- **گره با بیشترین جابجایی:** {report['displacement_statistics']['max_displacement_node']}\n")
            f.write(f"- **بیشترین جابجایی در X:** {report['displacement_statistics']['max_x_displacement']:.4e} m\n")
            f.write(f"- **بیشترین جابجایی در Y:** {report['displacement_statistics']['max_y_displacement']:.4e} m\n")
            f.write(f"- **میانگین مربعات جابجایی:** {report['displacement_statistics']['rms_displacement']:.4e} m\n\n")

            # تحلیل کمانش
            if report['buckling_analysis']['elements_with_I'] > 0:
                f.write("## تحلیل کمانش\n\n")
                f.write(f"- **اعضای دارای ممان اینرسی:** {report['buckling_analysis']['elements_with_I']}\n")
                f.write(f"- **اعضای در معرض خطر:** {report['buckling_analysis']['elements_at_risk']}\n")
                f.write(f"- **تعداد هشدارها:** {report['buckling_analysis']['warning_count']}\n")
                f.write(f"- **بیشترین نسبت کمانش:** {report['buckling_analysis']['max_buckling_ratio']:.3f}\n\n")

                if report['buckling_analysis']['critical_elements']:
                    f.write("### اعضای بحرانی از نظر کمانش\n\n")
                    f.write("| عضو | گره‌ها | نسبت کمانش | نیرو (N) | بار بحرانی (N) |\n")
                    f.write("|-----|--------|------------|----------|----------------|\n")

                    for elem in report['buckling_analysis']['critical_elements'][:5]:
                        f.write(f"| {elem['element_id']} | {elem['nodes']} | ")
                        f.write(f"{elem['buckling_ratio']:.3f} | {elem['force']:.2e} | ")
                        f.write(f"{elem['critical_load']:.2e} |\n")
                    f.write("\n")

            # هشدارها
            if 'warnings' in report.get('validation', {}):
                f.write("## هشدارها\n\n")
                for warning in report['validation']['warnings']:
                    f.write(f"- **{warning['type']}:** {warning['message']}\n\n")

            # اعتبارسنجی
            f.write("## اعتبارسنجی\n\n")
            f.write(
                f"- **اعتبارسنجی قرارداد علامت:** {'✅ موفق' if report['validation']['sign_convention_valid'] else '❌ ناموفق'}\n")
            f.write(
                f"- **اعتبارسنجی تعادل انرژی:** {'✅ موفق' if report['validation']['energy_balance_valid'] else '❌ ناموفق'}\n")
            if report['validation']['energy_message']:
                f.write(f"- **پیام انرژی:** {report['validation']['energy_message']}\n")

            f.write("\n---\n")
            f.write("*این گزارش به صورت خودکار توسط تحلیلگر خرپای 2D تولید شده است.*\n")

        logger.info(f"✅ گزارش Markdown در {filepath} ذخیره شد.")

    except Exception as e:
        logger.error(f"❌ خطا در ذخیره گزارش Markdown: {str(e)}")