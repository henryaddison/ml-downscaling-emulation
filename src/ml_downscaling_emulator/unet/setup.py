import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unet",
    version="0.0.1",
    author="",
    author_email="",
    description="U-Net implementation in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "unet"},
    packages=setuptools.find_packages(where="unet"),
    python_requires=">=3.6",
)
