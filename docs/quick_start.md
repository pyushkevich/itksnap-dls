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

### 1. Create a virtual environment

This step is not strictly necessary, but creating a virtual environment for the ITK-SNAP DLS server keeps its Python configuration separate. You can use the `conda` virtual environment manager, or other virtual environment tools. Here is the example using `conda`:

```
conda create -n itksnap-dls python=3.12
conda activate itksnap-dls
```

### 2. Install ITK-SNAP DLS

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

The first time you run this command, [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) models will be downloaded. After downloading the models, a list of URLs for the ITK-SNAP DLS service will be provided. 

```
$ python -m itksnap_dls 
************ ITK-SNAP Deep Learning Extensions Server ************
Use one of the following URLs to access the server from ITK-SNAP:
    http://127.0.1.1:8911
    http://10.150.40.44:8911
******************************************************************
```

Connecting to DLS from ITK-SNAP
--------------------------------

The deep learniing segmentation extension in ITK-SNAP is accessed under the paintbrush tool, when activating the "AI" button. The first time you press this button, the following window will appear:

![Dialog Window](images/itksnap_dls_dialog.png)

Pressing "Yes, configure" will open the Preferences Dialog, "AI Extensions" page, where you can enter the URL from Step 2. If successful, you will see a green "Connected" message under server status.

![Config window](images/dlsconfig.png)

If you get a red error message instead, please see troubleshooting below.

Using nnInteractive from ITK-SNAP
--------------------------------

* Just draw with the "AI" paintbrush. Your scribbles will be converted by **nnInteractive** to 3D segmentations. You can use the left mouse button to label pixels that should belong to the structure of interest, and right mouse button to label pixels that should be removed from the structure. When you change the active label, the **nnInteractive** interaction state is reset -- as if you are starting a new segmentation. 

![Interaction example](images/interaction.png)

Troubleshooting
---------------

### Errors running itksnap-dls on the server

1. Error messages related to SSL certificates (`SSL: CERTIFICATE_VERIFY_FAILED`):

    * Use `-k` option to disable certificate verification: 
    
            python -m itksnap_dls -k

2. `ERROR:    [Errno 13] error while attempting to bind on address ('0.0.0.0', 8891): permission denied`:

    * Use `-p` option to specify a different port number. Check with you system administrator for what ports are available.

                python -m itksnap_dls -p 9233


### ITK-SNAP unable to connect to the server

Errors are likely caused by a firewall on your server. Some options to resolve this are:

* If `itksnap-dls` prints out more than one URL that can be used, try them all. URLs like `http://127.0.0.1:8911` should be used when ITK-SNAP and `itksnap-dls` are running on the same machine. Other URLs, like `http://10.150.40.44:8911` are meant to be used when `itksnap-dls` is running on a remote machine. 

* Ask the system administrator to open the port being used by `itksnap-dls`, e.g., port 8911 or whatever port number you provide using the `-p` option. 

* Use secure shell (SSH) tunneling. This feature is provided by the command-line program `ssh.exe` (Windows) or `ssh` (MacOS/Linux). Open a terminal window on the computer running ITK-SNAP, and run the command

        ssh username@servername -L 8891:localhost:8891

    For example

        ssh pauly@10.150.40.44 -L 8891:localhost:8891

    As long as the SSH session is active, all traffic to your local machine's port 8891 will be forwarded to the server. In ITK-SNAP, you would now use the URK `http://localhost:8891` to connect to the server.


