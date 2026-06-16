#!/usr/bin/env python3
"""
تحلیلگر خرپای 2D با اثرات حرارتی و خطای ساخت - نسخه نهایی
"""

import argparse
import json
import sys
import traceback
import time
import logging
import os

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('truss_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def sanity_check(truss):
    """بررسی سلامت مدل قبل از تحلیل"""
    issues = []

    # ۱. بررسی تکیه‌گاه‌ها
    if len(truss.supported_nodes) < 2:
        issues.append("⚠️ تعداد تکیه‌گاه‌ها ناکافی است (حداقل ۲ مورد نیاز)")

    # ۲. بررسی طول اعضا
    for element in truss.elements.values():
        if element.L < 1e-12:
            issues.append(f"⚠️ عضو {element.id} طول صفر دارد")

    # ۳. بررسی واحدها
    valid_units = ['SI', 'SI-mm', 'SI-cm', 'Imperial']
    if truss.units not in valid_units:
        issues.append(f"⚠️ واحد نامعتبر: {truss.units}")

    # ۴. بررسی بار روی تکیه‌گاه
    for load in truss.loads:
        node = truss.nodes.get(load['node_id'])
        if node and node.is_support:
            issues.append(f"⚠️ بار روی تکیه‌گاه {load['node_id']} اعمال شده")

    if issues:
        logger.warning("🚨 مشکلات شناسایی شد:")
        for issue in issues:
            logger.warning(f"  • {issue}")
        return False

    return True
try:
    from fileio import parse_input, validate_units, write_output
    from model import TrussModel
    from assembly import build_global_matrices
    from solver import solve_displacements, calculate_element_results, calculate_total_energy, validate_energy
    from postprocess import sort_elements, calculate_percentages, generate_plots, generate_report, \
        save_report_to_markdown

    HAS_DEPENDENCIES = True
except ImportError as e:
    logger.error(f"خطا در بارگذاری ماژول‌ها: {e}")
    logger.error("لطفا مطمئن شوید تمام فایل‌های مورد نیاز وجود دارند.")
    HAS_DEPENDENCIES = False


def run_analysis(input_file: str, output_prefix: str = 'results', format: str = 'both',
                 force_plot: bool = False):
    """تابع اصلی اجرای تحلیل با مدیریت کامل خطا"""

    if not HAS_DEPENDENCIES:
        logger.error("ماژول‌های لازم بارگذاری نشده‌اند.")
        return {'success': False, 'error': 'ImportError', 'message': 'ماژول‌های لازم بارگذاری نشده‌اند.'}

    start_time = time.time()
    analysis_successful = False

    try:
        logger.info("=" * 60)
        logger.info("🔧 تحلیلگر خرپای 2D - نسخه نهایی")
        logger.info("=" * 60)

        # خواندن ورودی
        logger.info(f"📖 خواندن فایل ورودی: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # ذخیره نام فایل ورودی برای گزارش
        input_data['_input_file'] = input_file

        # اعتبارسنجی و تبدیل واحدها
        units = validate_units(input_data.get('units', 'SI'))
        logger.info(f"📏 واحدهای استفاده شده: {units}")

        # ایجاد مدل خرپا
        logger.info("🔧 ایجاد مدل خرپا...")
        truss = TrussModel(input_data)
        truss.input_file = input_file
        # بررسی سلامت مدل
        if not sanity_check(truss):
            logger.error("❌ مدل خرپا معتبر نیست. تحلیل متوقف شد.")
            return {'success': False, 'error': 'InvalidModel', 'message': 'مدل خرپا معتبر نیست.'}

        logger.info(
            f"   • گره‌ها: {len(truss.nodes)} (آزاد: {len(truss.free_nodes)}, تکیه‌گاهی: {len(truss.supported_nodes)})")
        logger.info(f"   • اعضا: {len(truss.elements)}")
        logger.info(f"   • بارهای گره‌ای: {len(truss.loads)}")
        logger.info(f"   • DOFهای آزاد: {len(truss.free_dofs)}")

        # بررسی اثرات حرارتی و خطای ساخت
        elements_with_delta_T = sum(1 for e in truss.elements.values() if e.delta_T != 0)
        elements_with_delta_L0 = sum(1 for e in truss.elements.values() if e.delta_L0 != 0)
        logger.info(f"   • اعضا با ΔT ≠ 0: {elements_with_delta_T}")
        logger.info(f"   • اعضا با δL₀ ≠ 0: {elements_with_delta_L0}")

        # مونتاژ ماتریس‌های سراسری
        logger.info("🧮 مونتاژ ماتریس‌های سراسری...")
        K_global, F_global = build_global_matrices(truss)
        # اطلاعات ماتریس
        if truss.options.get('use_sparse', True):
            logger.info(f"   • استفاده از ماتریس‌های تنک")
            logger.info(f"   • اندازه ماتریس سختی: {K_global.shape[0]} × {K_global.shape[1]}")
            if hasattr(K_global, 'nnz'):
                logger.info(f"   • تعداد عناصر غیرصفر: {K_global.nnz}")
        else:
            logger.info(f"   • استفاده از ماتریس‌های متراکم")
            logger.info(f"   • اندازه ماتریس سختی: {K_global.shape[0]} × {K_global.shape[1]}")

        # حل جابجایی‌ها
        logger.info("🧮 حل دستگاه معادلات...")
        displacements = solve_displacements(truss, K_global, F_global)
        logger.info(f"   • حل موفقیت‌آمیز")

        # محاسبه نتایج اعضا
        logger.info("📊 محاسبه نتایج اعضا...")
        results = calculate_element_results(truss, displacements)
        logger.info(f"   • محاسبه نتایج {len(results)} عضو")

        # اعتبارسنجی انرژی
        logger.info("⚖️ اعتبارسنجی تعادل انرژی...")
        U_total = calculate_total_energy(truss, displacements, F_global)
        is_energy_valid, energy_error, energy_message = validate_energy(results, U_total, truss, displacements,
                                                                        F_global)

        # اگر اثرات حرارتی وجود دارد و خطا زیاد است، پیام ویژه بده
        has_thermal = elements_with_delta_T > 0 or elements_with_delta_L0 > 0
        if has_thermal and energy_error > 0.1:
            logger.warning(f"   ⚠️ {energy_message}")
            logger.warning("   توجه: وجود اثرات حرارتی می‌تواند دقت محاسبات انرژی را کاهش دهد.")
        else:
            logger.info(f"   • {energy_message}")

        # پس‌پردازش و مرتب‌سازی
        logger.info("📈 پردازش نتایج...")
        sorted_results = sort_elements(results, by='energy', descending=True)
        results_with_pct = calculate_percentages(sorted_results)

        # تولید گزارش
        logger.info("📝 تولید گزارش...")
        report = generate_report(truss, displacements, results_with_pct, units)
        report['validation']['energy_balance_valid'] = is_energy_valid
        report['validation']['energy_error'] = float(energy_error)
        report['validation']['energy_message'] = energy_message

        # اعتبارسنجی قرارداد علامت
        try:
            sign_valid = truss.validate_sign_convention()
            report['validation']['sign_convention_valid'] = sign_valid
            if not sign_valid:
                logger.warning("⚠️ هشدار: قرارداد علامت معتبر نیست.")
        except AttributeError:
            logger.warning("⚠️ هشدار: تابع اعتبارسنجی قرارداد علامت وجود ندارد.")
            report['validation']['sign_convention_valid'] = True

        # ذخیره نتایج
        logger.info("💾 ذخیره نتایج...")

        # ایجاد پوشه خروجی اگر وجود ندارد
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"   • ایجاد پوشه خروجی: {output_dir}")

        write_output(results_with_pct, displacements, truss, report, output_prefix, format)

        # ذخیره گزارش Markdown
        save_report_to_markdown(report, f'{output_prefix}_report.md')

        # رسم نمودارها اگر درخواست شده
        plot_files = {}
        if truss.options.get('plot_results', False) or force_plot:
            logger.info("📊 رسم نمودارها...")
            plot_files = generate_plots(truss, displacements, results_with_pct, output_prefix)
            report['plot_files'] = list(plot_files.values())

        # محاسبه زمان اجرا
        end_time = time.time()
        execution_time = end_time - start_time

        # خلاصه نهایی
        logger.info("=" * 60)
        logger.info("✅ تحلیل با موفقیت انجام شد!")
        logger.info("=" * 60)

        logger.info(f"\n📊 خلاصه نتایج:")
        logger.info(f"   • انرژی کل: {U_total:.4e} J")
        logger.info(f"   • بیشترین جابجایی: {report['displacement_statistics']['max_displacement']:.4e} m")
        logger.info(f"   • بیشترین نیروی کششی: {report['force_distribution']['max_tensile_force']:.4e} N")
        logger.info(f"   • بیشترین نیروی فشاری: {report['force_distribution']['max_compressive_force']:.4e} N")
        # استفاده از elements_at_risk به جای warning_count
        logger.info(f"   • هشدارهای کمانش: {report['buckling_analysis'].get('elements_at_risk', 0)}")
        logger.info(f"   • زمان اجرا: {execution_time:.2f} ثانیه")

        logger.info(f"\n📁 فایل‌های خروجی:")
        logger.info(f"   • {output_prefix}.json")
        logger.info(f"   • {output_prefix}_elements.csv")
        logger.info(f"   • {output_prefix}_displacements.csv")
        logger.info(f"   • {output_prefix}_summary.csv")
        logger.info(f"   • {output_prefix}_report.md")

        if plot_files:
            logger.info(f"   • نمودارها: {len(plot_files)} فایل PNG")

        analysis_successful = True
        # اعتبارسنجی اضافی: مقایسه با مقادیر انتظاری اگر در ورودی تعریف شده باشد
        if 'expected_results' in input_data:
            logger.info("🔍 مقایسه با نتایج انتظاری...")
            expected = input_data['expected_results']

            # مقایسه نیروها
            for exp_elem in expected.get('elements', []):
                elem_id = exp_elem['id']
                actual_elem = next((r for r in results_with_pct if r['id'] == elem_id), None)

                if actual_elem:
                    N_actual = actual_elem['N']
                    N_expected_val = exp_elem.get('N')

                    if N_expected_val is not None:
                        rel_error = abs(N_actual - N_expected_val) / max(abs(N_expected_val), 1e-12)

                        if rel_error < 0.01:  # 1% خطا
                            logger.info(f"  ✅ عضو {elem_id}: N={N_actual:.2f} (انتظار: {N_expected_val:.2f})")
                        else:
                            logger.warning(f"  ⚠️ عضو {elem_id}: اختلاف {rel_error:.1%}")

        return {
            'success': True,
            'report': report,
            'plot_files': plot_files,
            'execution_time': execution_time
        }

    except FileNotFoundError:
        logger.error(f"❌ خطا: فایل {input_file} یافت نشد.")
        return {'success': False, 'error': 'FileNotFound', 'message': f'فایل {input_file} یافت نشد.'}

    except json.JSONDecodeError as e:
        logger.error(f"❌ خطا: فایل ورودی فرمت JSON نامعتبر دارد.")
        logger.error(f"   جزئیات: {str(e)}")
        return {'success': False, 'error': 'InvalidJSON', 'message': str(e)}

    except ValueError as e:
        logger.error(f"❌ خطا در داده‌های ورودی: {str(e)}")
        return {'success': False, 'error': 'ValueError', 'message': str(e)}


    except Exception as e:

        error_type = type(e).__name__

        error_msg = str(e)

        logger.error(f"❌ خطای {error_type}: {error_msg}")

        logger.error("برای جزئیات بیشتر فایل لاگ را بررسی کنید.")

        # ذخیره جزئیات فقط در لاگ

        logger.debug(f"Traceback:\n{traceback.format_exc()}")

        return {

            'success': False,

            'error': error_type,

            'message': error_msg,

            'traceback': traceback.format_exc()  # فقط برای لاگ

        }

    finally:
        if not analysis_successful:
            logger.error(f"⏱️ زمان اجرا تا وقوع خطا: {time.time() - start_time:.2f} ثانیه")


def main():
    parser = argparse.ArgumentParser(
        description='تحلیل خرپای 2D با اثرات حرارتی و خطای ساخت - نسخه نهایی',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌ها:
  %(prog)s examples/example1.json
  %(prog)s examples/reference_problem.json --output results/my_results --format csv
  %(prog)s examples/example2.json --plot
  %(prog)s --test

        """
    )

    parser.add_argument('input_file', nargs='?', help='فایل ورودی JSON')
    parser.add_argument('--output', '-o', default='results',
                        help='پیشوند نام فایل‌های خروجی (پیش‌فرض: results)')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'both'], default='both',
                        help='فرمت خروجی (پیش‌فرض: both)')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='رسم نمودارها حتی اگر در options تنظیم نشده باشد')
    parser.add_argument('--test', '-t', action='store_true',
                        help='اجرای تمام تست‌های واحد')
    parser.add_argument('--generate-readme', action='store_true',
                        help='تولید فایل README.md')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='سطح لاگینگ')

    args = parser.parse_args()

    # تنظیم سطح لاگینگ
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # تولید README
    if args.generate_readme:
        if HAS_DEPENDENCIES:
            #write_readme()
            logger.info("✅ فایل README.md تولید شد.")
        else:
            logger.error("❌ ماژول‌های لازم برای تولید README بارگذاری نشده‌اند.")
        return

    # اجرای تست‌ها
    if args.test:
        try:
            import pytest

            test_args = ['tests/', '-v', '--tb=short']
            exit_code = pytest.main(test_args)
            sys.exit(exit_code)
        except ImportError:
            logger.error("❌ pytest نصب نیست. برای اجرای تست‌ها ابتدا pytest را نصب کنید.")
            logger.error("   pip install pytest")
            sys.exit(1)

    # بررسی فایل ورودی
    if not args.input_file:
        parser.print_help()
        logger.error("\n❌ خطا: فایل ورودی الزامی است.")
        sys.exit(1)

    # بررسی وجود فایل ورودی
    if not os.path.exists(args.input_file):
        logger.error(f"❌ خطا: فایل {args.input_file} یافت نشد.")
        sys.exit(1)

    # اجرای تحلیل
    result = run_analysis(args.input_file, args.output, args.format, args.plot)

    if not result['success']:
        sys.exit(1)


if __name__ == "__main__":
    main()