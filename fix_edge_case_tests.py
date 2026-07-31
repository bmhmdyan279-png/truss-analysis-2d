# fix_edge_case_tests.py

path = "tests/test_edge_cases.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

# اصلاح test_singular_matrix_supports
old_test1 = '''    # بررسی که سیستم می‌تواند تشخیص دهد ماتریس منفرد است
    from truss_analysis.assembly import build_global_matrices
    K, F = build_global_matrices(model)

    # rank باید کمتر از اندازه ماتریس باشد
    rank = np.linalg.matrix_rank(K)
    assert rank < K.shape[0], "ماتریس باید منفرد باشد"'''

new_test1 = '''    # بررسی که سیستم می‌تواند تشخیص دهد ماتریس منفرد است
    from truss_analysis.assembly import build_global_matrices
    K, F = build_global_matrices(model)

    # تبدیل sparse به dense (اگر sparse باشد)
    try:
        K_dense = K.toarray()
    except AttributeError:
        K_dense = np.array(K)

    # rank باید کمتر از اندازه ماتریس باشد
    rank = np.linalg.matrix_rank(K_dense)
    assert rank < K_dense.shape[0], "ماتریس باید منفرد باشد"'''

content = content.replace(old_test1, new_test1)

# اصلاح test_extreme_loads
old_test2 = '''    # بررسی overflow رخ نمی‌دهد
    from truss_analysis.assembly import build_global_matrices
    K, F = build_global_matrices(model)
    assert np.all(np.isfinite(K)), "ماتریس سختی نباید NaN/Inf داشته باشد"
    assert np.all(np.isfinite(F)), "بردار بار نباید NaN/Inf داشته باشد"'''

new_test2 = '''    # بررسی overflow رخ نمی‌دهد
    from truss_analysis.assembly import build_global_matrices
    K, F = build_global_matrices(model)

    # تبدیل sparse به dense
    try:
        K_dense = K.toarray()
    except AttributeError:
        K_dense = np.array(K)

    try:
        F_dense = F.toarray().flatten()
    except AttributeError:
        F_dense = np.array(F).flatten()

    assert np.all(np.isfinite(K_dense)), "ماتریس سختی نباید NaN/Inf داشته باشد"
    assert np.all(np.isfinite(F_dense)), "بردار بار نباید NaN/Inf نداشته باشد"'''

content = content.replace(old_test2, new_test2)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ test_edge_cases.py fixed - sparse matrix handling added")
