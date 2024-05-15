# DEEPFAKE-COMPTETITION-PAV

Repositório com implementações de algortimos e técnicas testados para competição de classificação de áudios fake e reais, disponibilizada no Kaggle para a disciplina de Processamento de Áudio e Voz, ministrada pelo professor Dr. Arlindo Galvão, semetre 2024/1.

## Time:

Gustavo dos Reis - gustavo.reis2@discente.ufg.br

Guilherme Henrique - guilherme_reis@discente.ufg.br

Isadora Mesquita - isadora.mesquita@discente.ufg.br

Evellyn Nicole - nicole@discente.ufg.br

## Problema

Com o avanço significativo na tecnologia de inteligência artificial, especialmente na área de transferência de voz e text-to-speech (TTS), os modelos estão se tornando cada vez mais eficientes em sintetizar a fala humana de maneira natural. No entanto, esse progresso traz consigo novos desafios, um dos quais é a detecção de fala sintética ou "fake".

Esse exercício é crucial, especialmente em um contexto em que a disseminação de informações através de áudio e vídeo é cada vez mais comum. Com o aumento do uso de tecnologias associadas à voz, há uma necessidade crescente de garantir a autenticidade e a integridade da comunicação. 

Dito isso, surge a necessidade de desenvolver modelos capazes de distinguir entre fala sintética ("fake") e fala humana autêntica ("real"). Esses modelos desempenham um papel crucial na identificação de conteúdo gerado por máquinas, como deepfakes de áudio, que podem ser usados para enganar, difamar ou manipular.

As soluções de detecção de fala sintética devem ser altamente robustas e precisas, com o objetivo de minimizar ao máximo os falsos positivos. Ou seja, evitar situações em que a fala humana autêntica seja erroneamente classificada como sintética, o que pode resultar em diversas consequências negativas, como censura injusta ou desconfiança na precisão dos sistemas de detecção.

Portanto, este repositório busca explorar diversas técnicas e modelos para abordar esse problema desafiador, com foco em desenvolver soluções que sejam não apenas precisas na identificação de fala sintética, mas também robustas o suficiente para garantir um baixo índice de falsos positivos.

## Modelos usados
Para realizar essa classificação foi testada uma gama de modelos. De uma CNN a um modelo transformer para classificação(wav2vec), a ideia é que esses algoritmos consigam através do áudio bruto ou de features extraídas, como o Mel Spectograma, aprender a realizar a classificação.

* CNN
* SVM
* Regressão Logística
* Resnet18 Feature Extractor
* Wav2vec2 like models - Fine Tuning - Feature Extractor


Entre algumas técnicas que foram utilizadas para melhorar a performance dos modelos, as que tiveram maior destaque:
* label smoothing
* data augmentation
* feature extraction
* transfer learning

O processo está melhor documentado no relátorio final da competição: [Relatório_PAV.pdf](Relatório_PAV.pdf).

## Códigos

Os códigos usados estão implementados pasta a pasta em API que permitem a flexibilização de experimentos.

NN/training - Códigos para treinar redes neurais e convolucionais e algortimos clássicos.

ResNet18-Extractor - Códigos para usar a Resnet18 como feature extractor.

W2V - Codigos para treinar e testar modelos wav2vec2 like.

## Resultados dos Modelos 

| Modelo                                                                         | F1   | KP   | kPriv |
| ------------------------------------------------------------------------------ | ---- | ---- | ----- |
| CNN                                                                            | 0.88 | 0,70 | 0,64  |
| SVM                                                                            | 0,85 | 0,60 | 0,66  |
| ResNet-18                                                                      | 0,90 | 0.76 | 0.71  |
| wav2vec base                                                                   | 0,98 | 4,41 | 5,14  |
| wav2vec base + ls 0.2                                                          | 0,98 | 0,57 | 0,5   |
| wav2vec2-large-xlsr-53-gender-recognition-librispeechxls                       | 0,98 | 0,57 | 0,6   |
| Fine tuning wav2vec2-large-xlsr-53-gender-recognition-librispeechxls + ls      | 0,96 | 0,38 | 0,45  |
| Fine tuning wav2vec2-large-xlsr-53-gender-recognition-librispeechxls + ls + dg | 0,94 | 0,36 | 0,28  |
| wav2vec2-large-xlsr-53-gender-recognition-librispeechxls + ls + dg + 10 epochs | 0,95 | 0,26 | 0,23  |

## Modelo Final

O melhor modelo para a competição foi o fine tuning do modelo https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech com label smoothing de 0.2 e data augmentation, disponivel em https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification . Os outros modelos e técnicas exploradas também apresentaram bons resultados.

Esse modelo foi o ganhador da competição com um score privado de 0.23 de loss

O modelo final foi testado no dataset [ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) que é um dataset para avaliar modelos nessa task e coseguiu atingir boas métricas:

| Métrica             | Valor  |
| ------------------- | ------ |
| Acurácia            | 0.9286 |
| Precisão            | 0.9999 |
| Revocação           | 0.9205 |
| F1-Score            | 0.9363 |
| Equal Error Rate    | 0.0401 |

### Amostras do conjunto de avaliação:

- real : 7355
- fake : 63882


