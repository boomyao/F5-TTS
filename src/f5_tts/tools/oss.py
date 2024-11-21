from minio import Minio
import os
from minio.commonconfig import CopySource

TMP_DIR = '/tmp'

minio_client = None
def get_minio_client():
    global minio_client
    if minio_client is None:
        minio_client = Minio(
            endpoint=f"{os.getenv('MINIO_HOST')}:{os.getenv('MINIO_PORT')}",
            secure=False,
            access_key=os.getenv('MINIO_ACCESS_KEY'),
            secret_key=os.getenv('MINIO_SECRET_KEY'),
        )
    return minio_client

def upload_file(file_path, file_name):
    get_minio_client().fput_object(os.getenv('MINIO_BUCKET'), file_name, file_path)

def get_file_url(file_name):
    return get_minio_client().presigned_get_object(os.getenv('MINIO_BUCKET'), file_name)

def clone_file(file_name, new_file_name):
    if (file_name == new_file_name):
        return file_name
    source = CopySource(os.getenv('MINIO_BUCKET'), file_name)
    get_minio_client().copy_object(os.getenv('MINIO_BUCKET'), new_file_name, source)
    return new_file_name

def cache_file(file_name, output_dir=TMP_DIR, ignore_object_dir=False):
    output_file_path = os.path.join(output_dir, file_name if not ignore_object_dir else os.path.basename(file_name))
    
    if os.path.exists(output_file_path):
        return output_file_path
    get_minio_client().fget_object(os.getenv('MINIO_BUCKET'), file_name, output_file_path)
    return output_file_path
