import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vcas",
    version="0.1.0",
    author="Ziteng Wang",
    author_email="wangzite23@mails.tsinghua.edu.cn",
    description="code implementation of VCAS: variance-controlled adaptive sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)