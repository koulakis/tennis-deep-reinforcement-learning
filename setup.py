import setuptools
from setuptools.command.install import install
import subprocess

UNITY_AGENTS_PATH = 'udacity_custom_unity_agents/'


class InstallUdacityCustomUnityAgents(install):
    """Install the agents defined by udacity before setting up the package."""
    def run(self):
        subprocess.run(f'pip install {UNITY_AGENTS_PATH}'.split(' '))
        install.run(self)


setuptools.setup(
    name="tennis",
    version="0.0.1",
    author="Marios Koulakis",
    description="This is a solution for the third project of the Udacity deep reinforcement learning course.",
    packages=['scripts'],
    install_requires=[
        'tensorflow==2.11.1',
        'mlagents',
        'numpy',
        'typer',
        'gym',
        'stable_baselines3'
    ],
    cmdclass={
        'install': InstallUdacityCustomUnityAgents
    },
    extras_require={
        'dev': [
            'jupyterlab',
            'flake8'
        ]
    },
    python_requires='~=3.6'
)
