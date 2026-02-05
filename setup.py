from setuptools import setup, find_packages

setup(
    name="image_classification",
    version="0.1.0",
    description="Advanced Image Classification with Modern Deep Learning Techniques",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "albumentations>=1.3.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
    ],
)
