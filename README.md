## System Preparations (Windows)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
1.	Install VisualStudio Code from (https://code.visualstudio.com/Download)
2.	Install WSL and make sure that it's upgraded to version 2 (https://learn.microsoft.com/de-de/windows/wsl/Install)
3.	Install ffmpeg on windows (https://www.geeksforgeeks.org/how-to-Install-ffmpeg-on-windows/)
4.	Install WSL Extension in Visual Studio Code (https://code.visualstudio.com/docs/remote/wsl)
5.	open Ubuntu Console via Powershell  directly from dropdown poweshell (Windos 11) OR type into console ubuntu
6.	Intall ffmpeg on ubuntu: sudo apt Install ffmpeg    
7.	Create virtual enviroment (https://packaging.python.org/en/latest/guides/Installing-using-pip-and-virtual-environments/): python3 -m venv .venv
8.	Activate virtual environment: source .venv/bin/activate
9.	Install pip: python3 -m pip Install
10.	Upgrade pip: python3 -m pip --version
11.	Install requirements.txt: pip Install -r requirements.txt
12.	If requirements doesn’t install use: pip Install *packagename*
13.	Open VS Code: code .
    
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Delete Current YOLO Model if Exists
   - Remove files like yolov8n.pt

2. Create New Folder for Framed Data
   - Name it something like frames

3. Run Script to Create YOLO Dataset
   - python labeling.py.py -l <Path-of-.json> -v <Path-of- video> -o <Path-of-the- output-folder> 
   

4. Edit Dataset Path in Ultralytics Configuration
   - Open settings.yaml in /home/.config/Ultralytics
   - Reference your newly created folder based on settings.yaml
   - Change your dataset.yamlaccording to your data

5. Train YOLO Model
   - python train_yolo.py
   

6. Validate YOLO Model
   - python3 3_yolo_inference.py /home/aymsadmin/workspaces/label_project/output.mp4-m n -t detect -p
   
7. Access the Video
  - Open the file " output.mp4" in VLC Player
  - You can see the labeled data in the video
--------------------------------------------------------------------------------------------------------------------------------------------------------------------



