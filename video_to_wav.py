import os
import subprocess

source_dir = "Actor_24"

destination_dir = "Audio_only/Song"
if not os.path.exists(os.path.join(destination_dir, source_dir)):
    os.makedirs(os.path.join(destination_dir, source_dir))

for file_name in os.listdir(source_dir):
    if file_name.split("-")[0] == "02":
        continue
    command = "ffmpeg -i " + source_dir + "/" + file_name + " -ab 160k -ac 2 -ar 44100 -vn " + destination_dir + "/" + source_dir + "/" + file_name[:-3] + "wav"
    # print(command)
    try:
        subprocess.call(command, shell=True)

    except ValueError:
        continue