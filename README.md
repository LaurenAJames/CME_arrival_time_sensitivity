# CME arrival time sensitivity: Ghost Fronts with HUXt

## Introduction
This repository contains the workings to the results presented in the journal article __Sensitivity of model estimates of CME propagation and arrival time to inner boundary conditions__ by James et al. (2022). This code has been developed in Python to model the CME event from December 12th 2008. The Ghost Front Theory, presented by [Scott et al, (2019)](https://doi.org/10.1029/2018SW002093), is represented using the HUXt solar wind model by [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3). 

## Installation
_Provided by the original HUXt download._

 ``HUXt`` is written in Python 3.7.3 and requires ``numpy``, ``scipy``, ``scikit-image``, ``matplotlib``, ``astropy``, ``sunpy``, ``h5py``, and ``moviepy v1.0.1``. Currently ``moviepy v1.0.1`` is not available on ``conda``, but can be downloaded from ``pip``. Additionally, to make animations, ``moviepy`` requires ``ffmpeg`` to be installed. Specific dependencies can be found in the ``requirements.txt`` and ``environment.yml`` files.

After cloning or downloading ``HUXt``, users should update [``code/config.dat``](code/config.dat) so that ``root`` points to the local directory where HUXt is installed.

The simplest way to work with ``HUXt`` in ``conda`` is to create its own environment. With the anaconda prompt, in the root directory of ``HUXt``, this can be done as:
```
>>conda env create -f environment.yml
>>conda activate huxt
``` 
Then deterministic and ensemble runs of HUXt can be accessed through 
```
>>jupyter lab code/ghostfronts.ipynb
```

## Useage
This work uses the HUXt solar wind model; a lightweight model descibed by [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3). Version 1.0.0 was downloaded and developed to include the functionality seen in later versions, such as running an ensemble. Tracking of the nose positon along the CME leading ege has also been included in this model's adaptations, therefore we can simulate data to explore the Ghost Front theory descibed by [Scott et al, (2019)](https://doi.org/10.1029/2018SW002093). 

This code has been developed to model the CME event from December 12th 2008. STEREO/SECCHI HI1 observations, ACE in-situ data, and solar wind speed soultions used in the work is included in the repository. 

Solar wind speed at the inner boundary of HUXt is required. The first method is user-defined. Here, we make use of the data assimulated BRaVDA solar wind scheme, descibed by [Lang et al. (2020)]( https://doi.org/10.1029/2018SW001857). The second method uses the [HelioMAS](https://doi.org/10.1029/2000JA000121) solution of Carrington Rotaion 2077, retrived from a [folder of boundary conditions](data/boundary_conditions).

The following Jupyter Notebooks are available in the repository:
* Computing the determinisitc and ensemble runs of HUXt can be done through [``ghostfronts.ipynb``](code/ghostfronts.ipynb)
* Analysis of the ensemble files can be accessed through [``EnsembleAnalysis.ipynb``](code/EnsembleAnalysis.ipynb)
* Plotting ACE in-stu data can be accessed through [``Plot Spacecraft data.ipynb``](code/Plot \Spacecraft \data.ipynb)
