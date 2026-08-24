import os
import subprocess

print("🚨 شروع پاکسازی فاجعه گیت...")

# 1. پاک کردن staging area
print("📦 پاک کردن staging area...")
subprocess.run(['git', 'reset'], check=False)

# 2. اصلاح .gitignore
print("📝 اصلاح .gitignore...")
gitignore_content = """
# Virtual Environment
.venv/
venv/
env/
ENV/
.venv

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so
*.dll
*.dylib

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
logs/
*.db
*.sqlite
"""

with open('.gitignore', 'w', encoding='utf-8') as f:
    f.write(gitignore_content)

# 3. حذف فایل‌های venv از tracking گیت (اگر قبلاً track شده‌اند)
print("🗑️ حذف .venv از گیت (بدون حذف از دیسک)...")
subprocess.run(['git', 'rm', '-r', '--cached', '.venv'], check=False,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 4. اضافه کردن فقط فایل‌های واقعی پروژه
print("✅ اضافه کردن فایل‌های پروژه...")
files_to_add = [
    '.gitignore',
    'README.md',
    'pyproject.toml',
    'main.py',
    'apply_phase0.py',
    'examples/',
    'src/',
    'tests/test_e2e_cli.py',
]

for f in files_to_add:
    if os.path.exists(f):
        subprocess.run(['git', 'add', f], check=False)

# 5. کامیت کردن
print("📝 ساخت کامیت...")
result = subprocess.run(
    ['git', 'commit', '-m', 'fix(phase-0): resolve critical crashes, align schemas, sync versions'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ کامیت با موفقیت ساخته شد!")
    print(result.stdout)
else:
    print("❌ خطا در کامیت:")
    print(result.stderr)
    exit(1)

# 6. پوش کردن
print("🚀 پوش به گیت‌هاب...")
subprocess.run(['git', 'push', 'origin', 'main'], check=True)

print("\n🎉 پاکسازی کامل شد! مخزن شما آماده فاز ۱ است.")