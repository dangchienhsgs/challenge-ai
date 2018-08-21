import os
import pandas as pd
from pliers.extractors import *
import sys
from tqdm import tqdm

hop_length = 16000

extractors = [
    STFTAudioExtractor(),
    SpectralCentroidExtractor(hop_length=hop_length),
    SpectralBandwidthExtractor(hop_length=hop_length),
    SpectralContrastExtractor(),
    SpectralRolloffExtractor(hop_length=hop_length),
    PolyFeaturesExtractor(),
    RMSEExtractor(hop_length=hop_length),
    ZeroCrossingRateExtractor(hop_length=hop_length),
    ChromaSTFTExtractor(),
    ChromaCQTExtractor(),
    ChromaCENSExtractor(),
    MelspectrogramExtractor(),
    MFCCExtractor(),
    TonnetzExtractor(hop_length=hop_length),
    TempogramExtractor()
]

labels = ['female_central',
          'male_south',
          'female_south',
          'male_north',
          'male_central',
          'female_north']

unusual_fields = {'order', 'duration', 'object_id', 'onset'}


class Extraction:

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def extract(self):
        files = os.listdir(self.base_dir)

        data = {'file_name': []}

        for file in tqdm(files):
            try:
                full_path = '{0}/{1}'.format(self.base_dir, file)

                # Audio is sampled at 11KHz; let's compute power in 1 sec bins

                for ext in extractors:
                    result = ext.transform(full_path).to_df()
                    metrics = [x for x in result.columns.values if x not in unusual_fields]

                    for metric in metrics:
                        dsc = result[metric].describe()

                        self.add_value(data, '{0}_std'.format(metric), dsc['std'])
                        self.add_value(data, '{0}_mean'.format(metric), dsc['mean'])
                        self.add_value(data, '{0}_25'.format(metric), dsc['25%'])
                        self.add_value(data, '{0}_50'.format(metric), dsc['50%'])
                        self.add_value(data, '{0}_75'.format(metric), dsc['75%'])

                data['file_name'].append(file)
            except Exception as e:
                print("Error file {0}".format(file))

        return pd.DataFrame(data)

    @staticmethod
    def add_value(data, field, value):
        if field not in data:
            data[field] = []

        data[field].append(value)


if __name__ == "__main__":
    label = sys.argv[1]

    df = Extraction('data/accent_gender/train/{0}'.format(label)).extract()
    df.to_csv('{0}_new_extract.csv'.format(label))
