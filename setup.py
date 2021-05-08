from setuptools import find_packages, setup

# load readme
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mnist-detection",
    version="0.1.7",
    author="Chenchao Zhao",
    author_email="chenchao.zhao@gmail.com",
    description="Simulated MNIST detection dataset",
    packages=find_packages(exclude=["tests", "scripts"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["torch", "numpy", "pillow", "omegaconf", "torchvision"],
    license="GNU",
)
