import os
import zipfile
import logging
import glob
from src.Processor import Processor
import numpy as np
import boto3
import subprocess

ACCESS_KEY = ''
SECRET_KEY = ''  #old ones not working anyways

logger = logging.getLogger('S1ProcessorLogger')
logging.basicConfig(level=logging.INFO)

class S1Processor(Processor):
    def __init__(self, zips_path, footprint):
        super(S1Processor, self).__init__(zips_path, footprint)
        logger.info('Instanciating S1 processor for S1 files in {0}'.format(self.zips_path))

        self.suffix = 'S1'
        self.dtype = np.float32
        self.safe_folders = []
        self.basenames = []
        self.pols = []
        self.polarizations = []
        
    def checkDEM(self):
        if not os.path.exists('./Mosaic_W_EUR.tif'):
            if os.path.exists('./Mosaic_W_EUR.zip'):
                logger.info('Must Extract the dem file from archive...')
                with zipfile.ZipFile('./Mosaic_W_EUR.zip', 'r') as f:
                    f.extractall('.')
            else:
                logger.info('Must download S3 file from S3 Bucket...')
                s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
                s3.download_file('demfiles', 'Mosaic_W_EUR.zip', 'Mosaic_W_EUR.zip')
                with zipfile.ZipFile('./Mosaic_W_EUR.zip', 'r') as f:
                    f.extractall('.')
    
    def unzip(self):

        for zip_file in glob.glob(self.zips_path + '/S1*.zip'):
                    
            basename = os.path.basename(zip_file)[:-4]
            self.basenames.append(basename)
            self.safe_folders.append(os.path.join(self.zips_path, basename) + '.SAFE')

            with zipfile.ZipFile(zip_file, 'r') as f:
                f.extractall(self.zips_path)   


    def get_meta(self):
        for i in range(len(self.safe_folders)):
            only_safe_folder = os.path.basename(self.safe_folders[i])
            modestamp = only_safe_folder.split("_")[1]
            productstamp = only_safe_folder.split("_")[2]
            polstamp = only_safe_folder.split("_")[3]
            polarization = polstamp[2:4]

            self.polarizations.append(polarization)

            if polarization == 'DV':
                self.pols.append('VH,VV')
            elif polarization == 'DH':
                self.pols.append('HH,HV')
            elif polarization == 'SH' or polarization == 'HH':
                self.pols.append('HH')
            elif polarization == 'SV':
                self.pols.append('VV')
            else:
                self.pols.append('NaN')
                logger.info("Polarization error!")


    def process(self):

        self.checkDEM()
        self.unzip()
        self.get_meta()

        for i, safe_folder in enumerate(self.safe_folders):
       
            pol = self.pols[i]
            polarization = self.polarizations[i]
            footprint = self.footprint
            output_path = os.path.join(self.zips_path, self.basenames[i]) + '_VV_VH_dB.tif'

            subprocess.call(['python', 'src/subp.py', safe_folder, pol, polarization, footprint, output_path])
            

        logger.info('merging now S1 scenes')
        self.paths_to_merge = glob.glob(os.path.join(self.zips_path, '*.tif'))
        self.merge()
        logger.info('Done merging S1')


