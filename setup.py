from setuptools import setup, find_packages

setup(
    name="lslm",
    version="0.1.0",
    description="Listening-while-Speaking Language Model Implementation",
    author="San",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.0.0",
        "torchaudio>=0.8.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "jax>=0.4.13",
        "tensorflow>=2.12.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ]
    },
    python_requires=">=3.7",
) 