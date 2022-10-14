from pathlib import Path
import numpy as np
import cv2
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from pydicom.pixel_data_handlers.util import apply_voi_lut

DATASET = 'test'
scan_types = ['FLAIR','T1w','T1wCE','T2w']
data_root = Path('/home/bono/trsna')


def get_image_plane(data):
    x1, y1, _, x2, y2, _ = [round(j) for j in data.ImageOrientationPatient]
    cords = [x1, y1, x2, y2]

    if cords == [1, 0, 0, 0]:
        return 'Coronal'
    elif cords == [1, 0, 0, 1]:
        return 'Axial'
    elif cords == [0, 1, 0, 0]:
        return 'Sagittal'
    else:
        return 'Unknown'


def get_voxel(study_id, scan_type):
    imgs = []
    dcm_dir = data_root.joinpath(DATASET, study_id, scan_type)
    dcm_paths = sorted(dcm_dir.glob("*.dcm"), key=lambda x: int(x.stem.split("-")[-1]))
    positions = []

    for dcm_path in dcm_paths:
        img = pydicom.dcmread(str(dcm_path))
        imgs.append(img.pixel_array)
        positions.append(img.ImagePositionPatient)

    plane = get_image_plane(img)
    voxel = np.stack(imgs)

    if plane == "Coronal":
        if positions[0][1] < positions[-1][1]:
            voxel = voxel[::-1]
            print(f"{study_id} {scan_type} {plane} reordered")
    elif plane == "Sagittal":
        if positions[0][0] < positions[-1][0]:
            voxel = voxel[::-1]
            print(f"{study_id} {scan_type} {plane} reordered")
    elif plane == "Axial":
        if positions[0][2] > positions[-1][2]:
            voxel = voxel[::-1]
            print(f"{study_id} {scan_type} {plane} reordered")
    else:
        raise ValueError(f"Unknown plane {plane}")
    return voxel


def normalize_contrast(voxel):
    if voxel.sum() == 0:
        return voxel
    voxel = voxel - np.min(voxel)
    voxel = voxel / np.max(voxel)
    voxel = (voxel * 255).astype(np.uint16)
    return voxel


def resize_voxel(voxel, sz=64):
    output = np.zeros((sz, sz, sz), dtype=np.uint16)

    if np.argmax(voxel.shape) == 0:
        for i, s in enumerate(np.linspace(0, voxel.shape[0] - 1, sz)):
            output[i] = cv2.resize(voxel[int(s)], (sz, sz))
    elif np.argmax(voxel.shape) == 1:
        for i, s in enumerate(np.linspace(0, voxel.shape[1] - 1, sz)):
            output[:, i] = cv2.resize(voxel[:, int(s)], (sz, sz))
    elif np.argmax(voxel.shape) == 2:
        for i, s in enumerate(np.linspace(0, voxel.shape[2] - 1, sz)):
            output[:, :, i] = cv2.resize(voxel[:, :, int(s)], (sz, sz))

    return output


for study_path in list(data_root.joinpath(DATASET).glob("*")) :
    study_id = study_path.name

    if not study_path.is_dir():
        continue


    for i, scan_type in enumerate(scan_types):
        voxel = get_voxel(study_id, scan_type)
        voxel = normalize_contrast(voxel)
        voxel = resize_voxel(voxel)

        fileMeta = pydicom.Dataset()
        fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
        fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = fileMeta

        ds.Rows = voxel.shape[0]
        ds.Columns = voxel.shape[1]
        ds.NumberOfFrames = voxel.shape[2]

        ds.PixelSpacing = [1, 1]
        ds.SliceThickness = 1
        ds.BitsAllocated = 16
        ds.PixelRepresentation = 1
        ds.PixelData = voxel.tobytes()

        ds.save_as(f'/home/bono/resized/64test/{study_id}-{scan_type}.dcm', write_like_original=False)
