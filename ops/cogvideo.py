import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/CogVideo'
sys.path.insert(0,reference)

from cogvideo_command import CogVideo
