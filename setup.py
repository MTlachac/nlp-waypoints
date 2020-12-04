from setuptools import setup

setup(name='nlp_waypoints',
      version='0.0.1',
      python_requires='~=3.7',
      install_requires=[
        'numpy',
        'torch',
        'pybullet',
        'gym',
        'stable_baselines',
        'numpy==1.19.2',
        'tensorflow==1.14.0',
        'pyquaternion'
      ]
)
