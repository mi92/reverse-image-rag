from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Run the playwright install command
        subprocess.check_call(['playwright', 'install'])

setup(
    name='rir_api',
    version='0.1.0',
    author='Michael Moor',
    author_email='',
    description='A reverse image search API for image captioning and visual question answering.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mi92/reverse-image-rag',
    packages=find_packages(),
    install_requires=[
        'playwright==1.41.2',
        'openai==1.12.0',
        'requests==2.31.0',
        'pandas==2.2.0',
        'numpy==1.26.4',
        'requests',
    ],
    python_requires='>=3.8',
    cmdclass={
        'install': PostInstallCommand,
    },
)


