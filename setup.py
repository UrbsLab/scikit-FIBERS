import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scikit-fibers",
    version="0.9.2",
    author="Harsh Bandhey",
    author_email="harsh.bandhey@cshs.org",
    description="A Scikit Learn compatible implementation of FIBERS Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UrbsLab/scikit-FIBERS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "lifelines",
        "skrebate==0.7",
        "matplotlib"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)
