from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='kinasenet',  # Required

    version='1.1.0',  # Required

    description='',  # Optional

    long_description=long_description,  # Optional

    long_description_content_type='text/markdown',  # Optional (see note above)

    url='https://github.com/TongjiZhanglab/ImSpiRE',  # Optional

    author='Yizhi Zhang',  # Optional

    author_email='yizhi_zhang@tongji.edu.com',  # Optional

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.10',
    ],

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.10',

    install_requires=[
    'numpy==1.22.4',
    'pandas==2.2.0',
    'scikit-learn==1.4.0',
    'tensorboard==2.15.1',
    'packaging==23.1',
    'pyarrow==15.0.0'
],

)