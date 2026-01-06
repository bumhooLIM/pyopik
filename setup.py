from setuptools import setup

setup(
    name="pyopik",
    version="0.1.0",
    description="Intrinsic collision probability calculator based on Dell'Oro & Paolicchi (1998)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/pyopik",
    
    # This tells setuptools that your source code is a single file named pyopik.py
    py_modules=["pyopik"],
    
    # Dependencies required to run the code
    install_requires=[
        "numpy>=1.18.0",
    ],
    
    # Metadata for PyPI
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires='>=3.6',
)