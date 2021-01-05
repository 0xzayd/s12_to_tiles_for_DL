from src.utils import _remove_bad_tiles, process_tile, extract_subs_npz, splitName, chunks, zipdir
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import glob
import os
from functools import partial
import zipfile
import boto3
import tqdm

ACCESS_KEY = 'XXXXXXXXXXXXXXXXXXXXX'
SECRET_KEY = 'XXXXXXXXXXXXXXXXXXXXX'

class NpzProcessor(object):
    def __init__(self, output_folder, output_city_folder):
        self.output_folder = output_folder
        self.output_city_folder = output_city_folder
        self.cpus = max(multiprocessing.cpu_count()-1,1)


    def make_tiles(self):

        if not os.path.exists(self.output_city_folder):
            os.mkdir(self.output_city_folder)
        
        for interval in glob.glob(self.output_folder + '/*'):
            interval_name = splitName(interval)

            s1_ = glob.glob(interval + '/*S1.tif')[0]
            stacked = glob.glob(interval + '/stacked.tif')[0]

            curr_dir = os.path.join(self.output_city_folder, interval_name)
            if not os.path.exists(curr_dir):
                os.mkdir(curr_dir)
            tiles_s1 = curr_dir + '/S1'
            tiles_s2 = curr_dir + '/S2'
            
            if not os.path.exists(tiles_s1):
                os.mkdir(tiles_s1)
            if not os.path.exists(tiles_s2):
                os.mkdir(tiles_s2)

            extract_subs_npz(s1_, tiles_s1, 512, 256)
            extract_subs_npz(stacked, tiles_s2, 512, 256)

        
    def check_tiles(self):
        s_folders = [r for r in glob.glob(self.output_city_folder + '/*/S*')]
        npz_files = [r for r in glob.glob(self.output_city_folder + '/*/S*/*.npz')]
        npz_files_per_folder = [glob.glob(r+ '/*.npz') for r in s_folders]

        assert [len(npz_files_per_folder[i]) == len(npz_files_per_folder[i+1]) for i in range(len(npz_files_per_folder)-1)] == [True for i in range(len(npz_files_per_folder)-1)]

        remove_bad_tiles = partial(_remove_bad_tiles, s_folders=s_folders)

        # sadly too big to be multiprocessed by py2.7
        # let's try it first and if it fails, just loop :(
        try:    
            p = Pool(self.cpus)
            for i, subnpzfiles in enumerate(tqdm.tqdm(chunks(npz_files, self.cpus))):
                print('checking {0} chunks of 5000 tile'.format(i+1))
                p.map(remove_bad_tiles, subnpzfiles)
            p.close()
            p.join()
        except OverflowError as error:
            for npz_file in tqdm.tqdm(npz_files):
                remove_bad_tiles(npz_file)

    def compress(self):
        self.archive_name = splitName(self.output_city_folder)
        #shutil.make_archive(archive_name, 'zip', self.output_city_folder) not gonna do it for large files
        zipf = zipfile.ZipFile(self.archive_name + '.zip', 'w', allowZip64=True)
        zipdir(self.output_city_folder, zipf)
        zipf.close()


    def upload(self):

        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

        s3.upload_file(self.archive_name + '.zip', 'sen12munich', self.archive_name+'.zip')
        print("Upload Successful")

    def process(self):
    	print('making tiles ...')
        self.make_tiles()
    	print('checking tiles...')
        self.check_tiles()
    	print('compressing....')
        self.compress()
    	print('uploading...')
        self.upload()

#must be useless after post_proc
#subfolders = glob.glob('*')
#list_1 = [splitName(r) for r in glob.glob(subfolders[0] + '/S1/*.npz')]
#list_2 = [splitName(r) for r in glob.glob(subfolders[1] + '/S1/*.npz')]

#diff1 = set(list_1)-set(list_2)
#diff2 = set(list_2)-set(list_1)
#diff = list(diff1)+list(diff2)

#for e in diff:
#    p1 = subfolders[0] + '/S1/' + e + '.npz'
#    p2 = subfolders[0] + '/S2/' + e + '.npz'
#    p3 = subfolders[1] + '/S1/' + e + '.npz'
#    p4 = subfolders[1] + '/S2/' + e + '.npz'

#    if os.path.exists(p1):
#        os.remove(p1)
#    if os.path.exists(p2):
#        os.remove(p2)
#    if os.path.exists(p3):
#        os.remove(p3)
#    if os.path.exists(p4):
#        os.remove(p4)

