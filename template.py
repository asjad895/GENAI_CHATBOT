import os,sys
from pathlib import Path
import os.path
import logging
while(True):
    project_name=input("enter project name: ")
    if project_name!='':
        break
list_of_files=[
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/entity/__init__.py",
    # f"{project_name}/intent/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"config/config.yaml",
    "schema.yaml",
    "requirements.txt",
    "setup.py",
    "main.py",
    "app.py",
    "logs.py",
    "exception.py"


]

for file_path in list_of_files:
    file_path=Path(file_path)
    filedir,file=os.path.split(file_path)
    if filedir!='':
        os.makedirs(filedir,exist_ok=True)
    if (not os.path.exists(file_path)or (os.path.getsize(file_path)==0)):
        with open(file_path,"w") as f:
            pass
    else:
        logging.info(f"file is already exist at: {file_path}")