from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time
import json
import numpy as np
import torch
import shutil
import subprocess

from ops.utils.general import *
from dataclasses import dataclass
import re


@dataclass
class UlogFilter:
    # None means * here which will match anything
    funcname: str | None
    name: str | None

    def getItem(self):
        return (self.funcname, self.name)
    
    def isequal(self, other):
        herelist = self.getItem()
        otherlist = other.getItem()
        for i, item in enumerate(herelist):
            if item is not None and item != otherlist[i]:
                return False
        return True

class Ulog:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            if not hasattr(cls, '_called_from_create'):
                raise TypeError("Ulog should be created by Ulog.create() method.")
            delattr(cls, "_called_from_create")
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, runname="tmp", description="No description.",rootdir="./cache/log", drop=False):
        if hasattr(self, "done_init"):
            return
        self.rootdir = rootdir

        self._delete_dropdir()

        max_try = 100
        for i in range(max_try):
            try:
                self._try_set_runname(runname, drop)
                break
            except Exception:
                delaytime = random.randint(1, 5)
                print("Ulog name conflict, hanging a random time. delay: ", delaytime)
                time.sleep(delaytime)

        self.json_path = os.path.join(self.workdir, "files.json")
        self.json_data = []

        self._add_record(runname, description, self.json_path)
        self.done_init = True
        self.filters = []

    def _try_set_runname(self, runname, drop):
        self.run_timestamp = self._get_timestamp()
        self.runname = runname + f"_{self.run_timestamp}"
        if drop:
            self.runname += "_drop"
        
        self.workdir = os.path.join(self.rootdir, self.runname)
        if os.path.exists(self.workdir):
            raise FileExistsError(f"Directory {self.workdir} already exists.")
        
        os.makedirs(self.workdir) # This may also raise?
        return True

    @classmethod
    def create(cls, runname="tmp", description="No description.", drop=False, rootdir="./cache/log"):
        if hasattr(cls, "_instance"):
            raise ValueError("Ulog already created.")
        cls._called_from_create = True
        return cls(runname, description, rootdir=rootdir, drop=drop)
    
    def _runname_time_exceeded(self, runname):
        # return True if time is 1 day ago
        pattern = r'\d{8}_\d{6}'
        matches = re.findall(pattern, runname)
        if len(matches) == 0:
            return False
        timestamp_part = matches[0]
        dirtime = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
        now = datetime.now()
        return (now - dirtime).days >= 1
    
    def _delete_dropdir(self):
        # if filedir ends with _drop, then delete it. To avoid deleting running dir in multi-processing, we delete dirs after 1day.
        for dir in os.listdir(self.rootdir):
            if dir.endswith("_drop"):
                if self._runname_time_exceeded(dir):
                    print(f"Deleting {dir}")
                    shutil.rmtree(os.path.join(self.rootdir, dir))


    def _get_timestamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _add_record(self, name, description, filepath, timestamp=None):
        record = {
            "name": name,
            "description": description,
            "filepath": filepath,
            "timestamp": self._get_timestamp() if timestamp is None else timestamp
        }
        self.json_data.append(record)

        with open(self.json_path, "w") as f:
            json.dump(self.json_data, f)
                    

    def _get_unique_filename(self, filename):
        base, ext = os.path.splitext(filename)
        counter = 1

        unique_filename = filename
        while os.path.exists(os.path.join(self.workdir, unique_filename)):
            unique_filename = f"{base}_{counter}{ext}"
            counter += 1

        return unique_filename
    
    def _get_default_record(self, ext, name="tmp", description="No description."):
        # ext = ".png"
        filename = self._get_unique_filename(name + ext)
        filepath = os.path.join(self.workdir, filename)
        return name, description, filepath
    
    def _copy_file(self, srcpath, dstpath):
        # if providing path, then just copy it
        if not os.path.exists(srcpath):
            raise FileNotFoundError(f"File {srcpath} not found.")
        shutil.copy(srcpath, dstpath)

    def install_filter(self, filter:UlogFilter):
        self.filters.append(filter)

        

    def will_add(self, funcname, args):
        local_condition = UlogFilter(funcname, args["name"])
        for filter in self.filters:
            if filter.isequal(local_condition):
                return False
        return True

        


    def add_img(self, img, name=None, description=None):
        '''
        Add image to the log.
        Parameters:
            img: path, torch.Tensor, np.ndarray, [HWC] or [CHW], for mask img, C = 1.
        '''
        if not self.will_add("add_img", locals()):
            return 
        name, description, filepath = self._get_default_record(".png", name, description)

        if isinstance(img, str):
            self._copy_file(img, filepath)
        else:
            img = to_numpy(img)
            img = to_HWC(img)
            img = infer_value_0_255(img)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            Image.fromarray(img).save(filepath)
        self._add_record(name, description, filepath)


    def add_traj(self, traj, name=None, description=None):
        '''
        Add trajs to the log.
        Parameters:
            traj: list of Mcam, [Mcam ...] or filepath (str).
        '''
        if not self.will_add("add_traj", locals()):
            return
        name, description, filepath = self._get_default_record(".pt", name, description)
        torch.save(traj, filepath)
        self._add_record(name, description, filepath)

    def add_video(self, video, name=None, description=None):
        if not self.will_add("add_video", locals()):
            return
        name, description, filepath = self._get_default_record(".mp4", name, description)

        if isinstance(video, str):
            self._copy_file(video, filepath)
        else:
            easy_save_video(video, filepath)
        self._add_record(name, description, filepath)

    def add_ply(self, plymgr, name=None, description=None):
        if not self.will_add("add_ply", locals()):
            return
        name, description, filepath = self._get_default_record(".ply", name, description)

        if isinstance(plymgr, str):
            self._copy_file(plymgr, filepath)
        else:
            plymgr.save_ply(filepath)
        self._add_record(name, description, filepath)

    def add_code(self, filepath_src, name=None, description=None):
        if not self.will_add("add_code", locals()):
            return
        valid_ext = [".py", ".ipynb", ".yaml"]
        name = os.path.splitext(os.path.basename(filepath_src))[0] if name is None else name
        ext_name = os.path.splitext(filepath_src)[1]
        if ext_name not in valid_ext:
            raise ValueError(f"Invalid code file extension {ext_name}.")
        name, description, filepath = self._get_default_record(ext_name, name, description)
        self._copy_file(filepath_src, filepath)
        self._add_record(name, description, filepath)

    def remove_all(self):
        # totally delete this run, any other call later will cause error
        shutil.rmtree(self.workdir)
        assert 0

    def remove(self, name=None, ext=None):
        if name is None and ext is None:
            raise ValueError("At least one of name and ext should be provided.")
        file_to_delete = []
        if name is not None and ext is not None:
            file_to_delete += [record for record in self.json_data if record["name"] == name and record["filepath"].endswith(ext)]
        elif name is not None:
            file_to_delete += [record for record in self.json_data if record["name"] == name]
        elif ext is not None:
            file_to_delete += [record for record in self.json_data if record["filepath"].endswith(ext)]
        
        for record in file_to_delete:
            os.remove(record["filepath"])
            self.json_data.remove(record)

class LogVisualizer():
    def __init__(self):
        self.root_path = "./cache/log"
        self.tb_path = "./cache/log/runs"
        self.exclude_dirs = ["runs"]

    def visualize_3dgs(self, logname:int|str, recsize = [0.1, 0.1]):
        extractor = LogExtractor().load_json(logname)



    def get_all_logs(self):
        '''
        sorted in timestamp order.
        '''
        logs = os.listdir(self.root_path)
        logs = [log for log in logs if log not in self.exclude_dirs]
        logs = self._sort_folders_by_timestamp(logs)
        return logs
    
    def print_all_logs(self):
        logs = self.get_all_logs()
        for i, log in enumerate(logs):
            print(f"{i}: {log}")
    
    def _sort_folders_by_timestamp(self, folder_names):
        def extract_timestamp(folder_name):
            try:
                timestamp_part = "_".join(folder_name.split("_")[-2:])
                return datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
            except:
                return None

        sorted_folders = sorted(folder_names, key=lambda x: extract_timestamp(x) or datetime.min, reverse=True)
        return sorted_folders

    def build_tb_one(self, logname:int|str):
        if isinstance(logname, int):
            logs = self.get_all_logs()
            if logname >= len(logs):
                raise ValueError(f"Log index {logname} out of range.")
            logname = logs[logname]
        elif isinstance(logname, str):
            if logname not in self.get_all_logs():
                raise ValueError(f"Log {logname} not found.")
        else:
            raise TypeError(f"Invalid logname type {logname}.")

        log_src_path = os.path.join(self.root_path, logname)
        writer = SummaryWriter(log_dir=self.tb_path)
        jsonpath = os.path.join(log_src_path, "files.json")
        with open(jsonpath, "r") as f:
            jsondata = json.load(f)
        count_dict = {}
        for record in jsondata:
            name = record["name"]
            filepath = record["filepath"]
            count_dict[name] = count_dict.get(name, -1) + 1
            if filepath.endswith(".png"):
                img = Image.open(filepath)
                writer.add_image(name, np.array(img), count_dict[name])
            elif filepath.endswith(".pt"):
                traj = torch.load(filepath)
                writer.add_video(name, traj, count_dict[name])
            elif filepath.endswith(".mp4"):
                vid = imageio.mimread(filepath)
                vid = torch.from_numpy(np.stack(vid))
                vid = einops.rearrange(vid, "T H W C -> 1 T C H W")
                writer.add_video(name, vid, count_dict[name])
            elif filepath.endswith(".ply"):
                writer.add_mesh(name, filepath, count_dict[name])
        writer.flush()


    
    def start_server(self):
        subprocess.run(f"tensorboard --logdir {self.tb_path} --bind_all --port 6006")

class LogExtractor:
     
    def __init__(self, root_path="./cache/log"):
        self.root_path = root_path
        self.tb_path = "./cache/log/runs"
        self.exclude_dirs = ["runs"]

    def try_load_first_log(self, runname, idx=0):
        logs = sorted(os.listdir(self.root_path))
        logs_time = {}
        for log in logs:
            name, time = self._extract_dirname(log)
            if name == runname:
                logs_time[log] = time
        if len(logs_time) <= 0:
            return False
        logs_time = sorted(logs_time.items(), key=lambda item: item[1])
        first_log = logs_time[idx][0]
        print("Loading from log: ", first_log)
        self.load_json(first_log)
        return True
    
    def _extract_dirname(self, dirname):
        pattern = r"^(.*?)_(\d{8}_\d{6})(?:_(.*))?$"

        match = re.match(pattern, dirname)
        if match:
            name = match.group(1)  
            time = match.group(2)  
            suffix = match.group(3)       
            return name, datetime.strptime(time, "%Y%m%d_%H%M%S")
        else:
            return None, None  


    def load_json(self, name):
        self.name = name
        log_src_path = os.path.join(self.root_path, name)
        jsonpath = os.path.join(log_src_path, "files.json")
        with open(jsonpath, "r") as f:
            self.jsondata = json.load(f)
        self.all_names = [record["name"] for record in self.jsondata]
        return self
    
    def get_resource_by_name(self, resource_name, ext="", idx=None):
        resource = []
        for record in self.jsondata:
            record["filepath"] = os.path.join(self.root_path, self.name, os.path.basename(record["filepath"]))
            if record["name"] == resource_name and record["filepath"].endswith(ext):
                if idx is not None:
                    if idx == 0:
                        return self.parse_resorce(record)
                    else:
                        idx -= 1
                        continue
                    
                resource.append(self.parse_resorce(record))
        if len(resource) == 0:
            raise ValueError(f"Resource {resource_name} not found. available: {self.all_names}")
        return resource
    
    def parse_resorce(self, record):
        filepath = record["filepath"]
        if filepath.endswith(".mp4"):
            return self._parse_mp4(record)
        elif filepath.endswith(".png"):
            return self._parse_img(record)
        elif filepath.endswith(".ply"):
            return self._parse_ply(record)
        elif filepath.endswith(".pt"):
            return torch.load(filepath)
        else:
            raise ValueError(f"File type not supported: {filepath}")
    
    def _parse_mp4(self, record):
        '''
        return a list of tensor of shape [H, W, C], range in 0-1
        '''
        filepath = record["filepath"]
        vid = imageio.mimread(filepath)
        vid = [torch.from_numpy(np.array(img)).float() / 255. for img in vid]
        return vid
    
    def _parse_img(self, record):
        '''
        return a img of shape [H, W, C], range in 0-1
        '''
        filepath = record["filepath"]
        img = Image.open(filepath)
        img = np.array(img) / 255.0
        return img
    
    
    def _parse_ply(self, record):
        '''
        return a plymgr
        '''
        from ops.gs.base import GaussianMgr
        filepath = record["filepath"]
        try:
            gs = GaussianMgr().load_ply(filepath)
        except:
            from ops.PcdMgr import PcdMgr
            gs = PcdMgr(ply_file_path=filepath)
        return gs
    
    def _parse_py(self, record):
        '''
        return a py file
        '''
        filepath = record["filepath"]
        with open(filepath, "r") as f:
            content = f.read()
        return content
            


if __name__ == "__main__":
    vis = LogVisualizer()
    vis.print_all_logs()