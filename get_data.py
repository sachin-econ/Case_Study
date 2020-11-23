import tarfile
import functions as fns

def obtain_data(source1, source2):

    data_sources = [source1, source2]
    try:
        for source in data_sources:
            url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/{}_crop.tar'.format(
                  source)
            filename = '{}_crop.tar'.format(source)
            print('Downloading {}...'.format(source))
            fns.download(url, filename)
            print('Complete')
    except url.DoesNotExist:
        print('Data Download Failed')

    try:
        for source in data_sources:
            print('Extracting {}...'.format(source))
            with tarfile.open('{}_crop.tar'.format(source), 'r') as f:
                f.extractall()
            print('Complete')
    except extract_file.DoesNotExist:
        print('Data Extraction Failed')
