# ZoomZoom Coach 

In the pursuit of finding tenths, we eventually find ourselves at the point where we feel we cannot possibly have found more time in a corner. This tool will attempt to provide telemetry data layered with a recorded lap to provide you with the data and analysis necessary to find PBs or be more consistent around the track. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. This project is currently being built so if you have any issues with installation please add it to the running list of issues.

### Prerequisites

You will need to install the following software at the minimum:
    - Project Cars 2
    - Python 2.7 w/ PIP
    - OBS Studio
    - Github

For further customization it is recommended to also install:
    - Microsoft Visual Studio

```
Here are links to example setup videos:
    Python w/ PIP: https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation
    Github: https://www.atlassian.com/git/tutorials/install-git#windows
    OBS Studio Installer: https://obsproject.com/wiki/OBS-Studio-Quickstart
    *Ensure to set hotkeys for starting/stopping recording to F9/F10. Otherwise you will need to get visual studio to change hotkeys
    Microsoft Visual Studio: https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2017
```

### Installing

Once minimum requirements are satisfied proceed with instructions below:

1. Clone/Download this repository in the desired directory

```
git clone https://github.com/gurmeetsidhu/ZoomZoomCoach.git
```
2. PIP install pipenv in order to download all necessary libraries

```
pip install -r requirements.txt
```
## Getting up and running to record data

1. Statup Project Cars 2 and ensure you are in game menu

2. Open OBS Studio (Ensure hotkeys are set to F9/F10 for starting/stopping recording. Set video file destination to "ScreenRecordings" folder) 

3. Run data recorder
```
./Debug/SMS_MemMapSample.exe
```
4. Return to game and start your race. Once you are completed or paused. Your data will be saved to ./Debug/mapLog.csv and recordings to ./Debug/ScreenRecordings/

5. Run MapVisualization.py to analyze your lap
```
python MapVisualization.py
```

## Authors

* **Gurmeet Sidhu** - *Initial work* - [GurmeetSidhu](https://github.com/gurmeetsidhu)

Other Contributors:
* **Patrick Alex** - *Supplementary Work/AC integration* - [pSwitchSkates](https://github.com/pswitchskates)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Hat tip to anyone who has actively used this tool and provided feedback