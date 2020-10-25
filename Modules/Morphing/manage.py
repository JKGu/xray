import os, shutil, pathlib, PIL, sys
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

def __create_folder_if_not_exist(str):
    if not os.path.isdir(str):
        try:
            os.mkdir(str)
        except OSError:
            print ('Failed to create '+ str +"\n")

__MYDIR = str(pathlib.Path(__file__).parent.absolute())
__USER = __MYDIR+ '/Morph/UserFiles/'
__INPUT = __MYDIR+'/Morph/UserFiles/InputImages/'
__OUTPUT = __MYDIR+'/Morph/UserFiles/OutputImages/'
__MASK = __MYDIR+'/Morph/UserFiles/Masks/'

def __cleanup_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def __build_wrkdir():
    __create_folder_if_not_exist(__USER)
    __create_folder_if_not_exist(__INPUT)
    __create_folder_if_not_exist(__MASK)
    __create_folder_if_not_exist(__OUTPUT)


def __cleanup_wrkdir():
    __cleanup_folder(__INPUT)
    __cleanup_folder(__OUTPUT)
    __cleanup_folder(__MASK)

def __copy_all_files(dir_src,dir_dst,suffix):
    files = os.listdir(dir_src)
    for file in files:
        if file.endswith(suffix):
            full_file_name = os.path.join(dir_src, file)
            shutil.copy(full_file_name,dir_dst)


def build_new_project(inputImagePath, customMaskPath = None):

    print('Building working directories...')
    __build_wrkdir()
    __cleanup_wrkdir()

    print('Loading images...')
    __copy_all_files(inputImagePath, __INPUT, '.png')
    inputs = os.listdir(__INPUT)

    if len(inputs) <= 1:
        print("WARNING: Insufficient images. Abort.")
        return

    print('Preparing the mask image...')
    if customMaskPath != None:
        print('Importing custom mask...')
        shutil.copy(customMaskPath,__MASK)
        os.rename(__MASK+os.listdir(__MASK)[0], __MASK+'mask.png')
        
    else:
        print('Generating an empty mask...')
        sampleName = inputs[0]
        image = PIL.Image.open(__INPUT+sampleName)
        width, height = image.size
        image = PIL.Image.new('RGB', (width,height))
        image.save(__MASK+"mask.png", "PNG")

    print('Validating image size...')
    standard = PIL.Image.open(__MASK+'mask.png')
    width, height = standard.size
    for img in inputs:
        image = PIL.Image.open(__INPUT+img)
        new_image = image.resize((width, height))
        new_image.save(__INPUT+img)

    print('New project successfully created.')

def start_project():
    dir_tmp = os.getcwd()
    os.chdir(__MYDIR+"\Morph")
    print("DIR changed to ",os.getcwd())
    print("Executing Example.exe...")
    os.system("Example.exe")
    print("Executable finished.")
    os.chdir(dir_tmp)
    print("DIR changed back to ",os.getcwd())
    print("Result image:")
    viewOutputImage()


def viewOutputImage():
    img = mpimg.imread(__OUTPUT+'output.png') 
    plt.imshow(img) 

def getOutputImage(outputPath):
    shutil.copy(__OUTPUT+'output.png', outputPath)
