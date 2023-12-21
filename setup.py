from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="OMADS",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='2312',
    packages=find_packages(include=['SLML', 'SLML.*']),
    description="Statistical Learning Models Library",
    install_requires=[
      'cocopp==2.6.3',
      'NOBM>=1.0.1',
      'numpy>=1.22.4',
      'pandas>=1.5.2',
      'scipy>=1.9.3',
      'setuptools>=58.1.0',
      'StatLML>=2.0.0'
    ],
    extras_require={
        'interactive': ['matplotlib>=3.5.2', 'plotly>=5.14.1'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Intended Audience :: Developers',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
  )