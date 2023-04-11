from setuptools import setup
setup(
    name='robust-triangulation-relaxations',
    version='1.0',
    author='Linus Härenstam-Nielsen',
    description='Globally optimal robust triangulation using semidefinite relaxations',
    url='https://github.com/Linusnie/robust-triangulation-relaxations',
    keywords='triangulation, robust, relaxations',
    python_requires='>=3.10',
    packages=['triangulation_relaxations'],
    install_requires=[
        'pandas',
        'tqdm',
        'numba',
        'cvxpy',
        'mosek',
        'tyro',
    ],
)