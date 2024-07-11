import pandas as pd
from util import setup_logger

LOG = setup_logger()


def load_data(data_path, config):
    # data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    # data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
    metadata_path = f"{data_path}/metadata.csv"
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    metadata_df = metadata_df.head(config['asr_training']['num_of_samples'])

    split = int(len(metadata_df) * 0.90)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    LOG.info(f"Size of the training set: {len(df_train)}")
    LOG.info(f"Size of the validation set: {len(df_val)}")

    return df_train, df_val