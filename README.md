# DEEPFAKE-COMPTETITION-PAV

Repositório com implementações de algortimos e técnicas testados para competição de classificação de áudios fake e reais, disponibilizada no Kaggle para a disciplina de Processamento de Áudio e Voz, ministrada pelo professor Dr. Arlindo Galvão, semetre 2024/1.

Time:

Gustavo dos Reis - gustavo.reis2@discente.ufg.br

Guilherme Henrique - guilherme_reis@discente.ufg.br

Isadora Mesquita - isadora.mesquita@discente.ufg.br

Evellyn Nicole - nicole@discente.ufg.br

## Problema

Com o surgimento de vários modelos de IA de transferência de voz e text-to-speech(TTS) com desempenho cada vez melhor em sintetizar a fala humana, surge a necessidade de modelos de IA capazes de identificar quando uma fala é sintética(fake) ou não(real). Esse repositório explora algumas técnicas e modelos para realizar essa classificação.

A natureza desse problema exige que soluções sejam robustas a erros e errem o mínimo de falso positivos possiveis.

## Modelos usados
Para realizar essa classificação foram testadas uma gama de modelos. De uma CNN a um modelo transformer para classificação(wav2vec), a ideia é que esses algoritmos consigam através do áudio bruto ou de features extraídas, como o Mel Spectograma, aprender a realizar a classificação.

* CNN
* SVM
* Regressão Logística
* Resnet18 Feature Extractor
* Wav2vec2 like models - Fine Tuning - Feature Extractor

Entre algumas técnicas que foram utilizadas para melhorar a performance dos modelos, estas tiveram maior destaque:
* label smoothing
* data augmentation
* feature extraction
* transfer learning

O processo está melhor documentado no relátorio final da competição: [Relatório_PAV.pdf](Relatório_PAV.pdf).

## Códigos

Os códigos usados estão implementados pasta a pasta em API que permitem a flexibilização de experimentos.

NN/training -> Códigos para treinar redes neurais e convolucionais e algortimos clássicos.

ResNet18-Extractor - Códigos para usar a Resnet18 como feature extractor.

W2V - Codigos para treinar e testar modelos wav2vec2 like. 
## Modelo Final

O melhor modelo para a competição foi o fine tuning do modelo https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech com label smoothing de 0.2 e data augmentation, disponivel em https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification . Os outros modelos e técnicas exploradas também apresentaram bons resultados.

Esse modelo foi o ganhador da competição com um score privado de 0.23 de loss

O modelo final foi testado no dataset [ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) que é um dataset para avaliar modelos nessa task e coseguiu atingir boas métricas:

### Amostras do conjunto de avalição:

* real : 7355

* fake : 63882

### Métricas: 

* Acuracia:0.9286

* Precisão:0.9999

* Recall:0.9205

* F1-Score:0.9363

* Equal Error Rate (EER):  0.0401
