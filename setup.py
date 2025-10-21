"""
Enhanced Setup Script for Credit Card Churn Prediction System
Created by: Rasith Novfal S
Purpose: Advanced package setup with additional dependencies and metadata
"""

from setuptools import find_packages, setup
from typing import List
import os

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced Credit Card Churn Prediction System with Ensemble Learning and Explainable AI"

def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements from file and return as list.
    
    Parameters
    ----------
    file_path : str
        Path to requirements file
        
    Returns
    -------
    List[str]
        List of package requirements
    """
    requirements = []
    hyphen_e_dot = '-e .'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            requirements = file.readlines()
            requirements = [req.strip() for req in requirements]
            requirements = [req for req in requirements if req and not req.startswith('#')]
            
            if hyphen_e_dot in requirements:
                requirements.remove(hyphen_e_dot)
                
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using minimal requirements.")
        requirements = [
            'pandas>=2.0.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.3.0',
            'lightgbm>=4.0.0',
            'Flask>=2.3.0',
            'plotly>=5.15.0',
            'shap>=0.42.0'
        ]
    
    return requirements

# Package metadata
PACKAGE_NAME = "enhanced-churn-prediction"
VERSION = "2.0.0"
AUTHOR = "Rasith Novfal S"
AUTHOR_EMAIL = "rasithnovfal@gmail.com"
DESCRIPTION = "Advanced Credit Card Churn Prediction with Ensemble Learning and Explainable AI"
URL = "https://github.com/rijasRasith/advanced-churn-prediction-system"

# Classification metadata
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Data Scientists",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Office/Business :: Financial",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]

# Keywords for PyPI
KEYWORDS = [
    "machine-learning", "churn-prediction", "credit-card", "ensemble-learning",
    "explainable-ai", "customer-analytics", "flask", "data-science",
    "feature-engineering", "model-interpretation"
]

# Entry points for CLI commands
ENTRY_POINTS = {
    'console_scripts': [
        'churn-predict=src.cli:main',
        'churn-train=src.pipeline.enhanced_train_pipeline:main',
    ],
}

# Additional package data
PACKAGE_DATA = {
    'src': [
        'templates/*.html',
        'static/css/*.css',
        'static/js/*.js',
        'static/images/*',
    ],
}

# Development dependencies
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0',
        'pre-commit>=3.0.0',
    ],
    'docs': [
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'sphinxcontrib-napoleon>=0.7',
    ],
    'viz': [
        'plotly>=5.15.0',
        'dash>=2.10.0',
        'bokeh>=3.0.0',
    ]
}

# Main setup configuration
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    python_requires=">=3.8",
    install_requires=get_requirements('enhanced_requirements.txt'),
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    zip_safe=False,
    
    # Project URLs
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": f"{URL}",
        "Documentation": f"{URL}#readme",
        "Funding": f"{URL}/sponsors",
    },
    
    # Additional metadata
    platforms=["any"],
    license="MIT",
    
    # Options for different build backends
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
)
