import sys
#import snappy
from snappy import ProductIO
from snappy import HashMap
from snappy import GPF

safe_folder = [1]
pol = sys.argv[2]
polarization = sys.argv[3]
footprint = sys.argv[4]
output_path = sys.argv[5]


logger = logging.getLogger('S1Sub')
logging.basicConfig(level=logging.INFO)

def apply_orbit_file(source):
    logger.info('\tApplying orbit file')
    parameters = HashMap()
    parameters.put('Apply-Orbit-File', True)
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output


def remove_thermal_noise(source):
    logger.info('\tThermal noise removal')
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output


def calibrate(source, pol, polarization):
    logger.info('\tCalibration')
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)

    if polarization == 'DH':
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV':
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH':
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV':
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        logger.info("Unknown polarization")

    parameters.put('selectedPolarisations', pol)
    parameters.put('outputImageScaleInDb', False)
    parameters.put('auxFile', 'Product Auxiliary File')
    parameters.put('outputImageInComplex', False)
    parameters.put('outputGammaBand', False)
    parameters.put('outputBetaBand', False)
    output = GPF.createProduct("Calibration", parameters, source)
    return output


def terrain_correction(source):
    logger.info('\tTerrain correction...')
    parameters = HashMap()
    parameters.put('demName', 'External DEM')
    parameters.put('externalDEMFile', './Mosaic_W_EUR.tif')
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    #parameters.put('mapProjection', 'AUTO:42001')       # comment this line if no need to convert to UTM/WGS84, default is WGS84
    parameters.put('saveProjectedLocalIncidenceAngle', True)
    parameters.put('saveSelectedSourceBand', True)
    parameters.put('nodataValueAtSea', True)
    parameters.put('saveSelectedSourceBand', True)
    parameters.put('incidenceAngleForSigma0', 'Use projected local incidence angle from DEM')
    
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output

def subset(source, footprint):
    logger.info('\tClipping to AOI')
    parameters = HashMap()
    parameters.put('geoRegion', footprint)
    output = GPF.createProduct('Subset', parameters, source)
    return output


def scale_db(source):
    logger.info('\tScaling to dB')
    parameters = HashMap()
    parameters.put('sourceBands', 'Sigma0_VV,Sigma0_VH')
    output = GPF.createProduct("LinearToFromdB", parameters, source)
    return output


def main(safe_folder, pol, polarization, footprint, output_path):
    scene = ProductIO.readProduct(safe_folder + '/manifest.safe')   
    applyorbit = apply_orbit_file(scene)
    thermaremoved = remove_thermal_noise(applyorbit)
    calibrated = calibrate(thermaremoved, pol, polarization)
    tercorrected = terrain_correction(calibrated)

    # subset here
    if footprint:
        tercorrected = subset(tercorrected, footprint)

    scaled_db = scale_db(tercorrected)
    
    print('writing')
    ProductIO.writeProduct(scaled_db, output_path, 'GeoTIFF-BigTIFF')
    print('finished writing')
    scene.dispose()
    scene.closeIO()


main(safe_folder, pol, polarization, footprint, output_path)
