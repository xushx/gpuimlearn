import setuptools

# 第三方依赖
requires = [
    "numpy", "scipy", "scikit-learn"
]

setuptools.setup(
    name="cpuimlearn",  # 包名称
    version="0.1",  # 包版本
    description="GPU-imLearn",  # 包详细描述
    long_description="CPU-imLearn",
    url="https://github.com/inbliz/gpuimLearn",   # 项目官网
    # packages=packages,    # 项目需要的包
    packages = setuptools.find_packages(),
    include_package_data=True,  # 是否需要导入静态数据文件
    python_requires=">=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3*",  # Python版本依赖
    install_requires=requires,  # 第三方库依赖
    zip_safe=False,  # 此项需要，否则卸载时报windows error
    classifiers=[    # 程序的所属分类列表
        'Programming Language :: Python :: 3'
    ],
)

