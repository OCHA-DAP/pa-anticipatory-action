import zipfile
import logging
import requests
import yaml
import coloredlogs
import json
from google.oauth2 import service_account
import pygsheets
import os
import logging
import io
from googleapiclient.http import MediaIoBaseDownload
from urllib.request import urlretrieve
import locale
import pandas as pd

logger = logging.getLogger(__name__)

def parse_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def config_logger(level="INFO"):
    # Colours selected from here:
    # http://humanfriendly.readthedocs.io/en/latest/_images/ansi-demo.png
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        field_styles={
            "name": {"color": 8},
            "asctime": {"color": 248},
            "levelname": {"color": 8, "bold": True},
        },
    )

def auth_googleapi():
    #Authenticate the google service account
    folderid_pa = '0AGYkOFcloQuyUk9PVA'
    gapi_auth = os.getenv('GAPI_AUTH')
    if not gapi_auth:
        logger.error("No authentication file found")
        return None
    try:
        info = json.loads(gapi_auth)
        scopes = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
        credentials = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        gclient = pygsheets.authorize(custom_credentials=credentials)
        #enable the predictive analytics folder, i.e. allow to write to that folder
        gclient.drive.enable_team_drive(folderid_pa)
        return gclient
    except Exception:
        logger.error("Couldn't authenticate")
        return None

def download_gdrive(gclient,fileid,output_file):
    request = gclient.drive.service.files().get_media(fileId=fileid)
    fh = io.FileIO(output_file, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

def unzip(zip_file_path, save_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

def download_ftp(url, save_path, logger_info=True):
    if logger_info:
        logger.info(f'Downloading "{url}" to "{save_path}"')
    urlretrieve(url, filename=save_path)

def download_url(url, save_path, chunk_size=128):
    # Remove file if already exists
    try:
        os.remove(save_path)
    except OSError:
        pass
    # Download
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def convert_to_numeric(df_col,zone="en_US"):
    if df_col.dtype == "object":
        locale.setlocale(locale.LC_NUMERIC, zone)
        df_col = df_col.apply(lambda x: locale.atof(x))
        df_col = pd.to_numeric(df_col, errors="coerce")
    return df_col


