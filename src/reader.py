import pandas as pd


class AudioDataSet:
    """
    Gender: Female: 0, Male: 1
    Accent: North: 0, Central: 1, South: 2
    """

    def __init__(self, path):
        self.path = path

        fc_df = pd.read_csv(f'{path}/new_female_central.csv', index_col=0)
        fn_df = pd.read_csv(f'{path}/new_female_north.csv', index_col=0)
        fs_df = pd.read_csv(f'{path}/new_female_south.csv', index_col=0)
        mc_df = pd.read_csv(f'{path}/new_male_central.csv', index_col=0)
        mn_df = pd.read_csv(f'{path}/new_male_north.csv', index_col=0)
        ms_df = pd.read_csv(f'{path}/new_male_south.csv', index_col=0)

        # setting label
        fc_df['gender_label'] = 0
        fn_df['gender_label'] = 0
        fs_df['gender_label'] = 0
        mc_df['gender_label'] = 1
        mn_df['gender_label'] = 1
        ms_df['gender_label'] = 1

        fc_df['accent_label'] = 1
        fn_df['accent_label'] = 0
        fs_df['accent_label'] = 2
        mc_df['accent_label'] = 1
        mn_df['accent_label'] = 0
        ms_df['accent_label'] = 2

        fc_df['sep_label'] = 0
        fn_df['sep_label'] = 1
        fs_df['sep_label'] = 2
        mc_df['sep_label'] = 3
        mn_df['sep_label'] = 4
        ms_df['sep_label'] = 5

        self.df = pd.concat([fc_df, fn_df, fs_df, mc_df, mn_df, ms_df])
        self.test_df = pd.read_csv(f'{path}/new_public_test.csv', index_col=0)


if __name__ == "__main__":
    ds = AudioDataSet('../data')
    print(ds.df.shape)
