from setuptools import setup, find_packages

setup(
    name="korea_travel_guide",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers",
        "datasets",
        "peft",
        "evaluate",
        "python-dotenv",
        "numpy",
        "wandb",
        # "scikit-learn",
        # "gradio",
        "ipykernel",
        "praw",
        "pandas",
        "beautifulsoup4",
        "lxml",
        "huggingface_hub",
        "nltk",
        "absl-py",
        "rouge-score",
        "bert_score"
    ],
)
