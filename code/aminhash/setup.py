from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='minhashh',
    ext_modules=cythonize(
        Extension(
            "minhash",
            sources=["minhash.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", '-march=native', '-ffast-math',
#'-unroll-count=4',
'-Wno-deprecated-declarations',
'-Wno-deprecated-api',
#'-mprefetchwt1'
                ],

            language='c++',
        ),
        annotate=True
    ),

    install_requires=["numpy"],
)
