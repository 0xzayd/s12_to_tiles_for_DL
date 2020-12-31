from src.utils import stack_rgbnir, _remove_bad_tiles, process_tile, extract_subs_npz, splitName
import multiprocessing
from multiprocessing import Pool
import glob
import os
from functools import partial

class NpzProcessor(object):
    def __init__(self, output_folder, output_city_folder):
        self.output_folder = output_folder
        self.output_city_folder = output_city_folder
        self.cpus = max(multiprocessing.cpu_count()-2,1)
        

    def stack(self):
        paths_to_bands = [(glob.glob(folder + '/*red*S2.tif')[0],glob.glob(folder + '/*green*S2.tif')[0],
                glob.glob(folder + '/*blue*S2.tif')[0],
                glob.glob(folder + '/*nir*S2.tif')[0]) for folder in glob.glob(self.output_folder + '/*')]

        p = Pool(self.cpus)
        p.map(stack_rgbnir, paths_to_bands)


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

            extract_subs_npz(s1_, tiles_s1, 72, 36)
            extract_subs_npz(stacked, tiles_s2, 72, 36)

        
    def check_tiles(self):
        s_folders = [r for r in glob.glob(self.output_city_folder + '/*/S*')]
        npz_files = [r for r in glob.glob(self.output_city_folder + '/*/S*/*.npz')]
        npz_files_per_folder = [glob.glob(r+ '/*.npz') for r in s_folders]

        assert [len(npz_files_per_folder[i]) == len(npz_files_per_folder[i+1]) for i in range(len(npz_files_per_folder)-1)] == [True for i in range(len(npz_files_per_folder)-1)]

        remove_bad_tiles = partial(_remove_bad_tiles, s_folders=s_folders)

        p = Pool(self.cpus)
        p.map(remove_bad_tiles, npz_files)
        p.close()



    def process(self):
        self.stack()
        self.make_tiles()
        self.check_tiles()