import os 
import math
import time
import zipfile
import requests
import numpy as np
import pandas as pd
from pydub import AudioSegment
from multiprocessing.pool import ThreadPool

def url_response(path_url_list):
    """
    Download file at url to a path on your computer
    """
    path, url = path_url_list
    r = requests.get(url, stream = True)
    with open(path, 'wb') as f:
        for ch in r:
            f.write(ch)
def make_path_and_url(clip_id):
    """
    For a Clip Id return the Path to download it locally and 
        the url where the file download is located
    """
    file_path = os.path.join("Raw_Capuchinbird_Clips",f"XC{clip_id} - Capuchinbird - Perissocephalus tricolor.mp3")
    url = f"https://xeno-canto.org/{clip_id}/download"
    return file_path, url

def download_and_unzip_sounds(id_url_list):
    """
    Downloads and Unzips into the correct folder all of the Raw Not Capuchin Clips 
        a user inputs in the id_url_list.
    """
    clip_id, url = id_url_list
    path = os.path.join("Raw_Not_Capuchinbird_Clips",f"{clip_id}.zip")
    destination_path = "Raw_Not_Capuchinbird_Clips"
    url_response((path,url))
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    os.remove(path)

def df_to_list_of_url_tuples(df):
    """
    Extracts a list of tuples from the provided Sound_Urls csv file
    """
    output = []
    ids = df["id"].tolist()
    urls = df["url"].tolist()
    for i in range(len(ids)):
        output.append((ids[i],urls[i]))
    return output

def parse_capuchinbird_clips(clip_tuple):
    """
    Parses the audio clips described by the clip_tuple into the Parsed_Capuchinbird_Clips folder
    """
    clip_id, starts_and_ends = clip_tuple
    ms_to_seconds = 1000
    mp3_filename = os.path.join("Raw_Capuchinbird_Clips",f"XC{clip_id} - Capuchinbird - Perissocephalus tricolor.mp3")
    sound = AudioSegment.from_mp3(mp3_filename)
    count = 0
    for start, end in starts_and_ends:
        sub_clip = sound[start*ms_to_seconds:end*ms_to_seconds]
        sub_clip_name = f"XC{clip_id}-{count}"
        sub_clip.export(os.path.join("Parsed_Capuchinbird_Clips",f"{sub_clip_name}.wav"), format="wav")
        count += 1

def parse_not_capuchinbird_clips(raw_file_name):
    raw_audio_folder_path = "Raw_Not_Capuchinbird_Clips"
    selected_clips_folder_path = "Parsed_Not_Capuchinbird_Clips"
    parsed_file_name = raw_file_name.split(".")[0]
    full_path = os.path.join(raw_audio_folder_path, raw_file_name)
    sound = AudioSegment.from_mp3(full_path)
    # Get Count of ~ 3 second clips from file
    ms_to_s = 1000
    num_clips = math.floor(len(sound)/ms_to_s)//3
    if num_clips>0:
        array_vals = np.array_split(np.array(range(len(sound))),num_clips)
        count = 0
        for clip in array_vals:
            start = int(clip[0])
            end = int(clip[-1])
            clip = sound[start:end]
            clip.export(os.path.join(selected_clips_folder_path,f"{parsed_file_name}-{count}.wav"), format="wav")
            count += 1
    else:
        sound.export(os.path.join(selected_clips_folder_path,f"{parsed_file_name}-0.wav"), format="wav")

def df_to_list_of_call_tuples(df):
    """
    Extracts a list of tuples from the provided Parsing_Single_Call_Timestamps csv file
    """
    output = []
    for clip_id in df["id"].unique():
        clip_df = df[df["id"]==clip_id].copy()
        starts = clip_df["start"].tolist()
        ends = clip_df["end"].tolist()
        clip_list = []
        for i in range(len(starts)):
            clip_list.append((starts[i],ends[i]))
        output.append((clip_id,clip_list))
    return output

def download_dataset():
    # create folder structure
    os.makedirs("Raw_Not_Capuchinbird_Clips", exist_ok=True)
    os.makedirs("Raw_Capuchinbird_Clips", exist_ok=True)
    os.makedirs("Parsed_Not_Capuchinbird_Clips", exist_ok=True)
    os.makedirs("Parsed_Capuchinbird_Clips", exist_ok=True)

    # Download Capuchinbird calls
    clip_ids = ['114131', '114132', '119294', '16803', '16804', '168899', '178167', '178168', '201990', '216010', '216012', 
            '22397', '227467', '227468', '227469', '227471', '27881', '27882', '307385', '336661', '3776', '387509', 
            '388470', '395129', '395130', '401294', '40355', '433953', '44070', '441733', '441734', '456236', '456314', 
            '46077', '46241', '479556', '493092', '495697', '504926', '504928', '513083', '520626', '526106', '574020', 
            '574021', '600460', '65195', '65196', '79965', '9221', '98557', '9892', '9893']
    paths_and_urls = list(map(make_path_and_url, clip_ids))
    ThreadPool(4).imap_unordered(url_response, paths_and_urls)
    # Download Other Sounds
    urls_df = pd.read_csv("Other_Sound_Urls.csv")
    url_list = df_to_list_of_url_tuples(urls_df)
    ThreadPool(4).imap_unordered(download_and_unzip_sounds, url_list)
    while(1):
        time.sleep(2)
        if len(os.listdir("Raw_Capuchinbird_Clips"))>=53:
            print("Finished Capuchinbird Call Download")
            break
    while(1):
        time.sleep(2)
        if len(os.listdir("Raw_Not_Capuchinbird_Clips"))>=33:
            print("Finished Other Sounds Download")
            break

def parse_datasets():
    calls_df = pd.read_csv("Parsing_Single_Call_Timestamps.csv")
    calls_list = df_to_list_of_call_tuples(calls_df)
    ThreadPool(4).imap_unordered(parse_capuchinbird_clips, calls_list)
    ThreadPool(4).imap_unordered(parse_not_capuchinbird_clips, os.listdir("Raw_Not_Capuchinbird_Clips"))
    while(1):
        time.sleep(2)
        if len(os.listdir("Parsed_Capuchinbird_Clips"))>=217:
            print("Finished Capuchinbird Call Parsing")
            break
    while(1):
        time.sleep(2)
        if len(os.listdir("Parsed_Not_Capuchinbird_Clips"))>=590:
            print("Finished Other Sounds Parsing")
            break