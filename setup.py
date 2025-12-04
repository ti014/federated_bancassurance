"""Setup script cho Vertical FL package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vertical-fl-bancassurance",
    version="1.0.0",
    author="Federated Learning Project",
    description="Vertical Federated Learning với SplitNN cho Lapse Prediction trong Bancassurance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "run-vertical-fl=scripts.run_vertical_fl:main",
        ],
    },
)
