from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="edgelight-gan",
    version="1.0.0",
    description="EdgeLight-GAN: An Intelligent Lightweight Multi-Resolution GAN for "
                 "Low-Light Image Enhancement with Edge-Aware Resource Control",
    author="T. Kavitha, A. Kathirvel",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    python_requires=">=3.9",
)
