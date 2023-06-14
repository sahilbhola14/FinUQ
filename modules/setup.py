from setuptools import setup, find_packages

setup(name="finuq",
      version="0.0.1",
      description="Finite Precision Uncertainty Quantification",
      author="Sahil Bhola",
      author_email="sbhola@umich.edu",
      install_requires=['numpy', 'scipy', 'matplotlib', 'torch'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False
      )
