from setuptools import setup, find_packages

setup(
    name='mytrees', 
    
    version='0.1.0',
    
    author='Sofia',
    author_email='sofiagrossi@gmail.com',
    
    description='lista4IA',
    
    packages=find_packages(),
    
    install_requires=[
        'numpy',
        'pandas',
        'graphviz' 
    ],
    
    keywords=['id3', 'c45', 'cart',],
    
    python_requires='>=3.8',
)
