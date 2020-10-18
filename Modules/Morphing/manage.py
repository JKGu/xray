import os
import shutil
import pathlib
def __create_folder_if_not_exist(str):
    if not os.path.isdir(str):
        try:
            os.mkdir(str)
        except OSError:
            print ('Failed to create '+ str +"\n")

__MYDIR = str(pathlib.Path(__file__).parent.absolute())

def build_wrkdir():
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/')
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/InputImages/')
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/OutputImages/')
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/Masks/')
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/tmp/')
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/tmp/tmp0/')
    __create_folder_if_not_exist(__MYDIR+ '/MorphLib/UserFiles/tmp/tmp1/')

def cleanup_tmpfolder():
    shutil.rmtree(__MYDIR+ '/MorphLib/UserFiles/tmp/')

def cleanup_wrkdir():
    cleanup_tmpfolder()
    shutil.rmtree(__MYDIR+ '/MorphLib/UserFiles/InputImages/')
    shutil.rmtree(__MYDIR+ '/MorphLib/UserFiles/OutputImages/')
    shutil.rmtree(__MYDIR+ '/MorphLib/UserFiles/Masks/')

def init_wrkdir(
    inputImagePath=__MYDIR+'/MorphLib/UserFiles/InputImages/',
    maskPath=__MYDIR+ '/MorphLib/UserFiles/Masks/'
    ):
    build_wrkdir()
    cleanup_tmpfolder()
    shutil.rmtree(__MYDIR+ '/MorphLib/UserFiles/OutputImages/')
    
    if inputImagePath != __MYDIR+'/MorphLib/UserFiles/InputImages/':
        images = os.listdir(inputImagePath)
        for file in images:
            if file.endswith('.png'):
                full_file_name = os.path.join(inputImagePath, file)
                shutil.copy(full_file_name,__MYDIR+ '/MorphLib/UserFiles/InputImages/')
    
    if maskPath != __MYDIR+ '/MorphLib/UserFiles/Masks/':
        masks = os.listdir(maskPath)
        for file in masks:
            if file.endswith('.png'):
                full_file_name = os.path.join(maskPath, file)
                shutil.copy(full_file_name,__MYDIR+ '/MorphLib/UserFiles/Masks/')


def getOutputImage(outputPath):
    shutil.copy(__MYDIR+'/MorphLib/UserFiles/OutputImages/output.png', outputPath)