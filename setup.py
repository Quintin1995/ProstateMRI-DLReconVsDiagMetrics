import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastMRI_PCa",
    version="0.0.1",
    author="Q.Y. van Lohuizen (rad)",
    author_email="q.y.van.lohuizen@umcg.nl",
    description="later",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    scripts=[
        ],
    python_requires='>=3.6',
)
