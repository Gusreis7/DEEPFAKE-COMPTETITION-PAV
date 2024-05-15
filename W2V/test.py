import os
import glob
import argparse
import sys
sys.path.append('utils') 
import tqdm
import pandas as pd
from sklearn import metrics

from omegaconf import OmegaConf
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)
from torch.utils.data import DataLoader
from utils import (
    DataColletorTest,
    preprocess_metadata,
    get_label_id, predict,
    save_conf_matrix
)
import json


def read_config(path):
    with open(path, "r") as arquivo:
        # Carrega os dados do arquivo JSON em um dicionário
        dados = json.load(arquivo)
    return dados


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='configs/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        '--model',
        type=str
    )
   
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    model_path = args.model
    model_config  =read_config(path=f'{model_path}/config.json')
    label2id = model_config['label2id']
    id2label = model_config['id2label']

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    metadata_paths = [
        "../audios/metadata/test.csv"
    ]

    dataset_names = [
        "teste"
    ]
    base_dir = ''


    for metadata_path, dataset_name in tqdm.tqdm(zip(metadata_paths, dataset_names)):
        print(f"Testing {dataset_name} ... \n")
        df = pd.read_csv(metadata_path)

        print("-"*50)

        test_dataset = preprocess_metadata(cfg=cfg, df=df)

        data_collator = DataColletorTest(
            processor=feature_extractor,
            sampling_rate=cfg.data.target_sampling_rate,
            padding=cfg.data.pad_audios,
            label2id=label2id,
            max_audio_len=cfg.data.max_audio_len
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=2,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=4
        )

        paths,preds, scores = predict(
            test_dataloader=test_dataloader,
            model=model,
            cfg=cfg
        )

        #df = df.replace({label_column: label2id})
        #df['pred_label']  = preds
        for i, path in enumerate(paths):
            paths[i] = os.path.basename(path)
        df_final = pd.DataFrame(columns=['id','fake_prob'])
        df_final['id'] = paths
        df_final['fake_prob'] = scores
        df_final.to_csv(f'test_{dataset_name}-by-{os.path.basename(model_path)}.csv',index = False)

if __name__ == '__main__':
    main()