# راهنمای مشارکت

از مشارکت شما استقبال می‌کنیم! لطفاً موارد زیر را رعایت کنید:

1. یک Issue برای تغییر مورد نظر ایجاد کنید.
2. مخزن را fork کنید و یک branch جدید با نام `feature/...` ایجاد کنید.
3. تغییرات را اعمال کرده و تست‌ها را با `pytest tests/` اجرا کنید.
4. Pull Request خود را با توضیح کامل ارسال کنید.

کد باید با [PEP 8](https://pep8.org/) سازگار باشد.


## 🗺️ Future Roadmap

The following features are planned for future releases:

### Short-term (v2.2.x)
- **Sparse Matrix Support**: Use `scipy.sparse` for large-scale truss analysis (>1000 nodes)
- **Load Combinations**: Support for multiple load cases and combinations (LRFD, ASD)
- **Dynamic Analysis**: Modal analysis and time-history analysis
- **GUI Interface**: Web-based GUI using Streamlit or FastAPI

### Medium-term (v3.x)
- **3D Truss Analysis**: Extend to three-dimensional truss structures
- **Non-linear Analysis**: Material non-linearity (plasticity) and geometric non-linearity (large deformations)
- **Optimization**: Size and topology optimization for minimum weight
- **Multi-physics**: Coupled thermal-structural analysis with temperature-dependent material properties

### Long-term (v4.x+)
- **Machine Learning Integration**: Predictive models for rapid analysis and design optimization
- **Cloud Computing**: Distributed computing for very large structures
- **Real-time Monitoring**: Integration with sensor data for structural health monitoring
- **Code Compliance**: Automated code checking (AISC, Eurocode, etc.)

## 🤝 How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass (`pytest`)
- Code coverage remains above 90% (`pytest --cov=src --cov-fail-under=90`)
- Pre-commit hooks pass (`pre-commit run --all-files`)
- Documentation is updated for new features
