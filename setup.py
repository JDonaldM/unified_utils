import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as dependencies:
    requirements = [pkg.strip() for pkg in dependencies]

setuptools.setup(
    name="unified_utlis",
    version="0.0.2",
    author="Jamie Donald-McCann",
    author_email="jamie.donald-mccann@port.ac.uk",
    description="Inference pipeline for analysis of unified multipoles.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JDonaldM/unified_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    install_requires=requirements,
    python_requires='>=3.7',
)