from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="OMADS",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='2408.0',
    packages=find_packages(include=['OMADS', 'OMADS.*']),
    description="Mesh Adaptive Direct Search (MADS)",
    install_requires=[
      'pandas',
      'setuptools>=58.1.0',
      'scipy',
      'pyDOE2',
      "pytest",
      'samplersLib>=2408',
      "pytest-cov"
    ],
    extras_require={
        'interactive': ['matplotlib>=3.5.2', 'plotly>=5.14.1'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Intended Audience :: Developers',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
  )