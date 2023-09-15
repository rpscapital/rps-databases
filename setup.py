import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = """
mysqlclient>=2.2.0
psycopg2-binary==2.9.7
python-dotenv>=0.17.1
SQLAlchemy==1.4.29
pandas==1.3.5"""

setuptools.setup(
    name="rps-databases",  # Replace with your own username
    version="0.2.3",
    author="Wilian Silva",
    author_email="wilianzilv@gmail.com",
    description="Funções para conexão com banco de dados",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpscapital/rps-databases",
    packages=setuptools.find_packages(),
    python_requires=">=3.5.2",
    install_requires=requirements,
)
