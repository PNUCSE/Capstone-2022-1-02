import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
from tqdm import tqdm


DATASET = ['train','test']
scan_types = ['FLAIR','T1w','T1wCE','T2w']
data_root = Path('/home/bono/trsna')


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def normalize_contrast(img):
    if img.sum() == 0:
        return img
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.float32)
    return img

for dataset in tqdm(DATASET):
    for study_path in list(data_root.joinpath(dataset).glob("*")) :
        study_id = study_path.name

        if not study_path.is_dir():
            continue

        for i, scan_type in enumerate(scan_types):

            dcm_dir = data_root.joinpath(dataset, study_id, scan_type)
            dcm_paths = sorted(dcm_dir.glob("*.dcm"), key=lambda x: int(x.stem.split("-")[-1]))
            j = 0

            for dcm_path in dcm_paths:
                img = pydicom.dcmread(str(dcm_path))
                img = img.pixel_array
                img = cv2.resize(img,(224,224))
                img = np.repeat(img[..., np.newaxis], 3, -1)
                img = normalize_contrast(img)

                createDirectory('/home/bono/resized_224/'+dataset)
                createDirectory('/home/bono/resized_224/'+dataset+'/'+study_id)
                createDirectory('/home/bono/resized_224/'+dataset+'/'+study_id+'/'+scan_type)
                cv2.imwrite(f'/home/bono/resized_224/{dataset}/{study_id}/{scan_type}/image-{j}.png', img)
                j = j+1
