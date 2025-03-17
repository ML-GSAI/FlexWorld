import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/mvdust3r'
sys.path.insert(0,reference)

from mvdust3r_command import MVDust3r
