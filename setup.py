import io
import re
import setuptools


with open('graphgps/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

setuptools.setup(
    name='graphgps',
    version=__version__,
    python_requires='>=3.10',
    install_requires=[
        'torch==2.6.0',
        'torch-scatter',
        'torch-sparse',
        'torch-geometric',
        'pytorch-lightning',
        'yacs',
        'torchmetrics',
        'performer-pytorch',
        'ogb'
    ],
    author='Yan Xiang',
    author_email='',
    description='This is a easy-to-use command line version of GraphGPS.',
    long_description=long_description,
    url='https://github.com/xiangyan93/graphgps4molalkit',
    packages=setuptools.find_packages(),
        entry_points={
        'console_scripts': [
            'graphgps_cv=graphgps.optuna.cross_validation:graphgps_cv',
            'graphgps_optuna=graphgps.optuna.optuna:graphgps_optuna',
        ]
    },
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=False,
)
