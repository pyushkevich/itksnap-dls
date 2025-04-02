ITK-SNAP DLS Quick Start
========================
ITK-SNAP DLS is in early development stages. Please bookmark this page and check for updates.

Overview
--------
This extension allows you to take advantage of powerful AI-based interactive segmentation tools provided by [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) directly from ITK-SNAP. In mere seconds, scribbles drawn using the ITK-SNAP paintbrush are transformed into complex 3D segmentations.

![ITK-SNAP DLS diagram](images/itksnap_dls.png)

Requirements
------------
A computer with an NVidia GPU and Python is required. This can be the same computer on which you run ITK-SNAP or another computer on your local network. We will refer to this computer as the "GPU server".

Installation on GPU Server
--------------------------
* Ensure that NVidia CUDA drivers are installed and up to date on the GPU server
* Ensure that Python is available on the GPU server

#### 1. Create a virtual environment

This step is not strictly necessary, but creating a virtual environment for the ITK-SNAP DLS server keeps its Python configuration separate. You can use the `conda` virtual environment manager, or other virtual environment tools. Here is the example using `conda`:

```
conda create -n itksnap-dls python=3.12
conda activate itksnap-dls
```

#### 2. Install ITK-SNAP DLS

Install ITK-SNAP DLS using `pip`:

```
pip install itksnap-dls
```

Running the ITK-SNAP DLS service
--------------------------------
To run the service with default options, run:

```
python -m itksnap_dls
```

The first time you run this command, [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) models will be downloaded. After downloading the models, a URL for the ITK-SNAP DLS service will be provided. 

* Copy this URL and paste it in ITK-SNAP, in the Preferences Dialog, "AI Extensions" page.



