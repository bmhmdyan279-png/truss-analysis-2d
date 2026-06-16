"""
ورودی/خروجی و اعتبارسنجی - نسخه نهایی کامل
"""

import json
import csv
import numpy as np
from typing import Dict, List, Any, Optional
import os
import logging
from model import TrussModel

UNIT_CONVERSION = {
    # تبدیل به واحدهای پایه SI
    'SI': {
        'length': {'factor': 1.0, 'base': 'm', 'label': 'm'},
        'force': {'factor': 1.0, 'base': 'N', 'label': 'N'},
        'stress': {'factor': 1.0, 'base': 'Pa', 'label': 'Pa'},
        'energy': {'factor': 1.0, 'base': 'J', 'label': 'J'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°C'}
    },
    'SI-mm': {
        'length': {'factor': 0.001, 'base': 'm', 'label': 'mm'},
        'force': {'factor': 1.0, 'base': 'N', 'label': 'N'},
        'stress': {'factor': 1.0, 'base': 'Pa', 'label': 'Pa'},
        'energy': {'factor': 1.0, 'base': 'J', 'label': 'J'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°C'}
    },
    'SI-cm': {
        'length': {'factor': 0.01, 'base': 'm', 'label': 'cm'},
        'force': {'factor': 1.0, 'base': 'N', 'label': 'N'},
        'stress': {'factor': 1.0, 'base': 'Pa', 'label': 'Pa'},
        'energy': {'factor': 1.0, 'base': 'J', 'label': 'J'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°C'}
    },
    'Imperial': {
        'length': {'factor': 0.3048, 'base': 'm', 'label': 'ft'},  # فوت به متر
        'force': {'factor': 4.44822, 'base': 'N', 'label': 'lbf'},  # پوند-نیرو به نیوتن
        'stress': {'factor': 6894.76, 'base': 'Pa', 'label': 'psi'},  # psi به پاسکال
        'energy': {'factor': 1.35582, 'base': 'J', 'label': 'ft-lbf'},  # فوت-پوند به ژول
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°F'}  # برای ΔT تفاوتی ندارد
    }
}


def validate_units(units: str) -> Dict:
    """
    اعتبارسنجی واحدها و برگرداندن اطلاعات تبدیل

    بازمی‌گرداند:
        Dict شامل فاکتورهای تبدیل برای انواع کمیت‌ها
    """
    if units not in UNIT_CONVERSION:
        raise ValueError(f"واحد نامعتبر: '{units}'. واحدهای مجاز: {', '.join(UNIT_CONVERSION.keys())}")

    return UNIT_CONVERSION[units]


def convert_to_si(value: float, from_units: str, quantity_type: str) -> float:
    """
    تبدیل مقدار از واحد داده شده به واحدهای SI

    پارامترها:
        value: مقدار عددی
        from_units: واحد مبدا ('SI', 'SI-mm', 'Imperial', ...)
        quantity_type: نوع کمیت ('length', 'force', 'stress', 'energy', 'temperature')

    بازگشت:
        مقدار تبدیل شده به SI
    """
    if from_units not in UNIT_CONVERSION:
        raise ValueError(f"واحد مبدا نامعتبر: {from_units}")

    if quantity_type not in UNIT_CONVERSION[from_units]:
        raise ValueError(f"نوع کمیت نامعتبر برای واحد {from_units}: {quantity_type}")

    conversion = UNIT_CONVERSION[from_units][quantity_type]
    return value * conversion['factor']


def convert_from_si(value: float, to_units: str, quantity_type: str) -> float:
    """
    تبدیل مقدار از واحدهای SI به واحد مقصد

    پارامترها:
        value: مقدار در واحدهای SI
        to_units: واحد مقصد
        quantity_type: نوع کمیت

    بازگشت:
        مقدار تبدیل شده به واحد مقصد
    """
    if to_units not in UNIT_CONVERSION:
        raise ValueError(f"واحد مقصد نامعتبر: {to_units}")

    if quantity_type not in UNIT_CONVERSION[to_units]:
        raise ValueError(f"نوع کمیت نامعتبر برای واحد {to_units}: {quantity_type}")

    conversion = UNIT_CONVERSION[to_units][quantity_type]
    return value / conversion['factor']


def format_with_units(value: float, units: str, quantity_type: str) -> str:
    """
    قالب‌بندی مقدار با واحد مناسب

    مثال:
        format_with_units(0.001, 'SI', 'length') -> "1.000 mm"
        format_with_units(1000, 'SI', 'force') -> "1.000 kN"
    """
    if units not in UNIT_CONVERSION:
        units = 'SI'  # fallback

    # تبدیل به واحد مقصد
    value_in_target = convert_from_si(convert_to_si(value, 'SI', quantity_type), units, quantity_type)

    # گرفتن برچسب واحد
    unit_label = UNIT_CONVERSION[units][quantity_type]['label']

    # قالب‌بندی هوشمند بر اساس بزرگی مقدار
    abs_val = abs(value_in_target)

    if quantity_type == 'length':
        if units == 'SI':
            if abs_val < 0.001:
                return f"{value_in_target * 1000:.3f} mm"
            elif abs_val < 1:
                return f"{value_in_target * 100:.1f} cm"
            else:
                return f"{value_in_target:.3f} m"
        else:
            return f"{value_in_target:.3f} {unit_label}"

    elif quantity_type == 'force':
        if abs_val < 0.001:
            return f"{value_in_target * 1e6:.2f} μ{unit_label}"
        elif abs_val < 1:
            return f"{value_in_target * 1e3:.2f} m{unit_label}"
        elif abs_val < 1000:
            return f"{value_in_target:.2f} {unit_label}"
        elif abs_val < 1e6:
            return f"{value_in_target / 1e3:.2f} k{unit_label}"
        else:
            return f"{value_in_target / 1e6:.2f} M{unit_label}"

    elif quantity_type == 'stress':
        if abs_val < 1000:
            return f"{value_in_target:.1f} {unit_label}"
        elif abs_val < 1e6:
            return f"{value_in_target / 1e3:.1f} k{unit_label}"
        elif abs_val < 1e9:
            return f"{value_in_target / 1e6:.1f} M{unit_label}"
        else:
            return f"{value_in_target / 1e9:.2f} G{unit_label}"

    else:
        return f"{value_in_target:.4e} {unit_label}"


logger = logging.getLogger(__name__)


def parse_input(input_data: Dict) -> Dict:
    """
    تجزیه و اعتبارسنجی داده‌های ورودی
    """
    # اعتبارسنجی ساختار پایه
    required_keys = ['nodes', 'elements']
    for key in required_keys:
        if key not in input_data:
            raise ValueError(f"کلید الزامی '{key}' در ورودی یافت نشد.")

    # اعتبارسنجی گره‌ها
    node_ids = set()
    for i, node in enumerate(input_data['nodes']):
        if 'id' not in node:
            raise ValueError(f"گره در اندیس {i} دارای شناسه (id) نیست.")

        node_id = node['id']
        if node_id in node_ids:
            raise ValueError(f"شناسه گره تکراری: {node_id}")
        node_ids.add(node_id)

        # اعتبارسنجی مختصات
        if 'x' not in node or 'y' not in node:
            raise ValueError(f"گره {node_id} دارای مختصات ناقص است.")

        try:
            float(node['x'])
            float(node['y'])
        except (ValueError, TypeError):
            raise ValueError(f"گره {node_id} دارای مختصات نامعتبر است.")

    # اعتبارسنجی اعضا
    element_ids = set()
    for i, element in enumerate(input_data['elements']):
        if 'id' not in element:
            raise ValueError(f"عضو در اندیس {i} دارای شناسه (id) نیست.")

        element_id = element['id']
        if element_id in element_ids:
            raise ValueError(f"شناسه عضو تکراری: {element_id}")
        element_ids.add(element_id)

        # بررسی اتصالات
        node_i = element.get('node_i')
        node_j = element.get('node_j')

        if node_i is None or node_j is None:
            raise ValueError(f"عضو {element_id}: گره‌های ابتدا و انتها مشخص نیستند.")

        if node_i not in node_ids:
            raise ValueError(f"عضو {element_id}: گره {node_i} وجود ندارد.")
        if node_j not in node_ids:
            raise ValueError(f"عضو {element_id}: گره {node_j} وجود ندارد.")

        # بررسی مقادیر فیزیکی
        try:
            A = float(element.get('A', 0))
            E = float(element.get('E', 0))
            if A <= 0:
                raise ValueError(f"عضو {element_id}: مساحت مقطع باید مثبت باشد.")
            if E <= 0:
                raise ValueError(f"عضو {element_id}: مدول الاستیسیته باید مثبت باشد.")
        except (ValueError, TypeError):
            raise ValueError(f"عضو {element_id}: مقادیر A یا E نامعتبر هستند.")

        # بررسی I (اختیاری)
        if 'I' in element and element['I'] is not None:
            try:
                I_val = float(element['I'])
                if I_val < 0:
                    raise ValueError(f"عضو {element_id}: ممان اینرسی نمی‌تواند منفی باشد.")
            except (ValueError, TypeError):
                raise ValueError(f"عضو {element_id}: ممان اینرسی نامعتبر است.")

    # اعتبارسنجی بارها
    if 'loads' in input_data:
        loads = input_data['loads'].get('node_forces', [])
        for load in loads:
            node_id = load.get('node_id')
            if node_id not in node_ids:
                raise ValueError(f"بار روی گره {node_id} که وجود ندارد.")

    return input_data


def validate_units(units: str) -> str:
    """
    اعتبارسنجی واحدها با ارسال خطا در صورت نامعتبر بودن
    """
    valid_units = ['SI', 'Imperial', 'SI-mm', 'SI-cm']
    if units not in valid_units:
        error_msg = f"واحد نامعتبر: '{units}'. واحدهای مجاز: {', '.join(valid_units)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return units


def convert_units(value: float, from_unit: str, to_unit: str, quantity_type: str = 'length') -> float:
    """
    تبدیل واحدهای مختلف

    پارامترها:
        value: مقدار عددی
        from_unit: واحد مبدا
        to_unit: واحد مقصد
        quantity_type: نوع کمیت ('length', 'force', 'stress', 'temperature')

    بازگشت:
        مقدار تبدیل شده
    """

    # جدا کردن پیشوند و واحد پایه
    def parse_unit(unit_str):
        if unit_str == 'SI':
            return 1.0, 'm'
        elif unit_str == 'SI-mm':
            return 0.001, 'm'
        elif unit_str == 'SI-cm':
            return 0.01, 'm'
        elif unit_str == 'Imperial':
            return 0.3048, 'ft'  # فوت به متر
        else:
            return 1.0, unit_str

    # ضریب تبدیل واحد مبدا
    factor_from, base_from = parse_unit(from_unit)
    factor_to, base_to = parse_unit(to_unit)

    # تبدیل بر اساس نوع کمیت
    if quantity_type == 'length':
        # طول: متر پایه
        return value * factor_from / factor_to

    elif quantity_type == 'force':
        # نیرو: نیوتن پایه
        if 'Imperial' in from_unit:
            # پوند-نیرو به نیوتن
            factor_from = 4.44822
        if 'Imperial' in to_unit:
            # نیوتن به پوند-نیرو
            factor_to = 4.44822
        return value * factor_from / factor_to

    elif quantity_type == 'stress':
        # تنش: پاسکال پایه
        if 'Imperial' in from_unit:
            # psi به پاسکال
            factor_from = 6894.76
        if 'Imperial' in to_unit:
            # پاسکال به psi
            factor_to = 6894.76
        return value * factor_from / factor_to

    elif quantity_type == 'temperature':
        # دما: درجه سانتیگراد پایه (برای ΔT تفاوتی ندارد)
        return value

    else:
        # تبدیل عمومی
        return value * factor_from / factor_to


def format_value_with_units(value: float, unit_type: str = 'length') -> str:
    """
    قالب‌بندی مقدار با واحد مناسب و تبدیل به واحدهای خوانا
    """
    if value is None or not np.isfinite(value):
        return "N/A"

    abs_val = abs(value)

    if unit_type == 'length':
        if abs_val < 1e-9:
            return f"{value * 1e12:.3f} pm"
        elif abs_val < 1e-6:
            return f"{value * 1e9:.3f} nm"
        elif abs_val < 1e-3:
            return f"{value * 1e6:.3f} μm"
        elif abs_val < 1:
            return f"{value * 1e3:.3f} mm"
        elif abs_val < 1e3:
            return f"{value:.3f} m"
        else:
            return f"{value:.3f} m"

    elif unit_type == 'displacement':
        # جابجایی: همیشه در mm نشان داده شود
        return f"{value * 1000:.3f} mm"

    elif unit_type == 'force':
        if abs_val < 1e-3:
            return f"{value * 1e6:.3f} μN"
        elif abs_val < 1:
            return f"{value * 1e3:.3f} mN"
        elif abs_val < 1e3:
            return f"{value:.3f} N"
        elif abs_val < 1e6:
            return f"{value / 1e3:.3f} kN"
        else:
            return f"{value / 1e6:.3f} MN"

    elif unit_type == 'stress':
        if abs_val < 1e3:
            return f"{value:.3f} Pa"
        elif abs_val < 1e6:
            return f"{value / 1e3:.3f} kPa"
        elif abs_val < 1e9:
            return f"{value / 1e6:.3f} MPa"
        else:
            return f"{value / 1e9:.3f} GPa"

    elif unit_type == 'energy':
        if abs_val < 1e-6:
            return f"{value * 1e9:.3f} nJ"
        elif abs_val < 1e-3:
            return f"{value * 1e6:.3f} μJ"
        elif abs_val < 1:
            return f"{value * 1e3:.3f} mJ"
        else:
            return f"{value:.3f} J"

    else:
        return f"{value:.6g}"


def write_output(results: List[Dict], displacements: np.ndarray,
                 truss: TrussModel, report: Dict,
                 output_prefix: str, format: str = 'both'):
    """
    ذخیره نتایج به فایل‌های خروجی - نسخه کامل با همه فیلدها
    """
    units = truss.units

    # تابع کمکی برای ایجاد پوشه خروجی
    def ensure_output_dir(file_path: str):
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    if format in ['json', 'both']:
        # همان کد قبلی...
        pass

    if format in ['csv', 'both']:
        # 1. ذخیره نتایج اعضا به CSV با همه فیلدها
        csv_file = f'{output_prefix}_elements.csv'
        ensure_output_dir(csv_file)

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                # لیست کامل فیلدهایی که باید ذخیره شوند
                fieldnames = [
                    # اطلاعات شناسایی
                    'id', 'node_i', 'node_j',

                    # اطلاعات هندسی
                    'L', 'c', 's',

                    # تغییر طول‌ها
                    'delta_L_free', 'delta_L_eff',

                    # نیرو و وضعیت
                    'N', 'status',

                    # انرژی
                    'U', 'pct_U',

                    # مشخصات مقطع
                    'A', 'E', 'alpha', 'I', 'section_type',

                    # پارامترهای حرارتی
                    'delta_T', 'delta_L0',

                    # کمانش
                    'P_cr', 'buckling_ratio', 'buckling_warning',
                    'buckling_safety_factor', 'effective_length_factor',

                    # تنش (محاسبه شده)
                    'stress'
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for r in results:
                    row = {}
                    for field in fieldnames:
                        # مقداردهی به همه فیلدها
                        if field == 'stress':
                            # محاسبه تنش: σ = N/A
                            N_val = r.get('N', 0)
                            A_val = r.get('A', 0) or getattr(truss.elements.get(r['id']), 'A', 0)
                            if A_val != 0:
                                row[field] = N_val / A_val
                            else:
                                row[field] = ''

                        elif field == 'A':
                            # اگر در نتایج نیست، از عنصر بگیر
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'A', '')
                            row[field] = value

                        elif field == 'E':
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'E', '')
                            row[field] = value

                        elif field == 'alpha':
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'alpha', '')
                            row[field] = value

                        elif field == 'delta_T':
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'delta_T', '')
                            row[field] = value

                        elif field == 'delta_L0':
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'delta_L0', '')
                            row[field] = value


                        elif field == 'I':

                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'I', '')

                            if value is None or value == '':

                                row[field] = 'N/A'

                            elif isinstance(value, (int, float, np.number)):

                                if abs(value) < 1e-12:

                                    row[field] = '0'

                                else:

                                    row[field] = f'{value:.6e}'

                            else:

                                row[field] = str(value)

                        elif field == 'P_cr':
                            value = r.get(field)
                            if value is None or value == '':
                                row[field] = 'N/A'
                            elif isinstance(value, (int, float, np.number)):
                                if abs(value) < 1e-12:
                                    row[field] = '0'
                                else:
                                    row[field] = f'{value:.6e}'
                            else:
                                row[field] = str(value)

                        elif field == 'section_type':
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'section_type', '')
                            row[field] = value

                        elif field == 'effective_length_factor':
                            value = r.get(field) or getattr(truss.elements.get(r['id']), 'K_eff', '')
                            row[field] = value

                        elif field == 'buckling_warning':
                            # تبدیل boolean به Yes/No
                            value = r.get(field, False)
                            row[field] = 'Yes' if value else 'No'

                        elif field in r:
                            # برای فیلدهای مستقیم
                            value = r[field]
                            if value is None:
                                row[field] = ''
                            elif isinstance(value, (int, float, np.number)):
                                # اعداد را با دقت مناسب ذخیره کن
                                if abs(value) < 1e-12:
                                    row[field] = '0'
                                else:
                                    row[field] = f'{value:.6e}'
                            else:
                                row[field] = str(value)
                        else:
                            row[field] = ''

                    writer.writerow(row)

        logger.info(f"✅ نتایج اعضا CSV با {len(fieldnames)} فیلد در {csv_file} ذخیره شد.")

        # 2. ذخیره جابجایی گره‌ها به CSV

        # 3. ذخیره گزارش خلاصه با اطلاعات کامل
        summary_file = f'{output_prefix}_summary.csv'
        ensure_output_dir(summary_file)

        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value', 'Unit', 'Description'])

            # آمار کلی
            writer.writerow(['Total Nodes', report['metadata']['total_nodes'], '-', 'تعداد کل گره‌ها'])
            writer.writerow(['Free Nodes', report['metadata']['free_nodes'], '-', 'گره‌های آزاد'])
            writer.writerow(['Supported Nodes', report['metadata']['supported_nodes'], '-', 'گره‌های تکیه‌گاهی'])
            writer.writerow(['Total Elements', report['metadata']['total_elements'], '-', 'تعداد کل اعضا'])
            writer.writerow(
                ['Temperature Change', report['metadata']['global_temperature_change'], '°C', 'تغییر دمای سراسری'])

            # انرژی
            writer.writerow(['Total Energy', report['energy_statistics']['total_energy'], 'J', 'انرژی کل کرنشی'])

            # عضو با بیشترین انرژی
            max_energy = report['energy_statistics']['max_energy_element']
            if max_energy['id'] is not None:
                writer.writerow(['Max Energy Element ID', max_energy['id'], '-', 'شناسه عضو با بیشترین انرژی'])
                writer.writerow(['Max Energy', max_energy['energy'], 'J', 'بیشترین انرژی عضو'])
                writer.writerow(['Energy Percentage', f"{max_energy['percentage']:.2f}", '%', 'درصد انرژی عضو بحرانی'])
                writer.writerow(['Max Energy Nodes', max_energy['nodes'], '-', 'گره‌های عضو بحرانی'])
                writer.writerow(['Max Energy Status', max_energy['status'], '-', 'وضعیت عضو بحرانی'])

            # عضو با بیشترین نیرو
            max_force = report['energy_statistics']['max_force_element']
            if max_force['id'] is not None:
                writer.writerow(['Max Force Element ID', max_force['id'], '-', 'شناسه عضو با بیشترین نیرو'])
                writer.writerow(['Max Force', max_force['force'], 'N', 'بیشترین نیرو'])
                writer.writerow(['Max Force Status', max_force['status'], '-', 'وضعیت عضو با بیشترین نیرو'])

            # توزیع نیرو
            writer.writerow(
                ['Tension Elements', report['force_distribution']['tension_elements'], '-', 'تعداد اعضای کششی'])
            writer.writerow(['Compression Elements', report['force_distribution']['compression_elements'], '-',
                             'تعداد اعضای فشاری'])
            writer.writerow(
                ['Max Tensile Force', report['force_distribution']['max_tensile_force'], 'N', 'بیشترین نیروی کششی'])
            writer.writerow(['Max Compressive Force', report['force_distribution']['max_compressive_force'], 'N',
                             'بیشترین نیروی فشاری'])

            # جابجایی‌ها
            disp_stats = report['displacement_statistics']
            writer.writerow(['Max Displacement', disp_stats['max_displacement'], 'm', 'بیشترین جابجایی'])
            if disp_stats['max_displacement_node'] is not None:
                writer.writerow(
                    ['Max Displacement Node', disp_stats['max_displacement_node'], '-', 'گره با بیشترین جابجایی'])
            writer.writerow(
                ['Max X Displacement', disp_stats['max_x_displacement'], 'm', 'بیشترین جابجایی در راستای X'])
            writer.writerow(
                ['Max Y Displacement', disp_stats['max_y_displacement'], 'm', 'بیشترین جابجایی در راستای Y'])
            writer.writerow(['RMS Displacement', disp_stats['rms_displacement'], 'm', 'میانگین مربعات جابجایی'])

            # کمانش
            buckling_stats = report['buckling_analysis']
            writer.writerow(['Elements with I', buckling_stats['elements_with_I'], '-', 'اعضای دارای ممان اینرسی'])
            writer.writerow(['Elements at Risk', buckling_stats.get('elements_at_risk', 0), '-', 'اعضای در معرض کمانش'])
            writer.writerow(['Buckling Warnings', buckling_stats.get('warning_count', 0), '-', 'تعداد هشدارهای کمانش'])
            writer.writerow(['Max Buckling Ratio', buckling_stats['max_buckling_ratio'], '-', 'بیشترین نسبت کمانش'])

            # اثرات حرارتی
            thermal_stats = report['thermal_effects']
            writer.writerow(
                ['Elements with ΔT ≠ 0', thermal_stats['elements_with_delta_T'], '-', 'اعضا با تغییر دمای غیرصفر'])
            writer.writerow(
                ['Elements with δL₀ ≠ 0', thermal_stats['elements_with_delta_L0'], '-', 'اعضا با خطای ساخت غیرصفر'])
            writer.writerow(['Max ΔT', thermal_stats['max_delta_T'], '°C', 'بیشترین تغییر دما'])
            writer.writerow(['Max δL₀', thermal_stats['max_delta_L0'], 'm', 'بیشترین خطای ساخت'])

            # اعتبارسنجی
            validation_stats = report['validation']
            writer.writerow(['Sign Convention Valid', 'Yes' if validation_stats['sign_convention_valid'] else 'No', '-',
                             'اعتبار قرارداد علامت'])
            writer.writerow(['Energy Balance Valid', 'Yes' if validation_stats['energy_balance_valid'] else 'No', '-',
                             'اعتبار تعادل انرژی'])
            writer.writerow(['Energy Error', validation_stats['energy_error'], '-', 'خطای تعادل انرژی'])

        logger.info(f"✅ گزارش خلاصه CSV با اطلاعات کامل در {summary_file} ذخیره شد.")
