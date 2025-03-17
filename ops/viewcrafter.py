import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/ViewCrafter'
sys.path.insert(0,reference)

from viewcrafter_command import ViewCrafter
