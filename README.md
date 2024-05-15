# DEEPFAKE-COMPTETITION-PAV

Repositorio com implementações de algortimos e técnicas testados para competição de classificação de audios fake e reais, lançada no Kaggle para a disciplina de Processamento de Audio e voz ministrada pelo professor Dr. Arlindo Galvão, semetre 2024/1.
Time:

Gustavo dos Reis - gustavo.reis2@discente.ufg.br

Guilherme Henrique - guilherme_reis@discente.ufg.br

Isadora Stefany - isadora.mesquita@discente.ufg.br

Evellyn Nicole - nicole@discente.ufg.br

## Problema

Com o surgimento de vários modelos de IA de transferencia de voz e text-to-speech(TTS) cada vez melhores em sinterizar a fala humana, surge a necessidade de modelos de IA capazes de identificar quando uma fala é sintetica(fake) ou não(real). Esse repositório explora algumas técnicas e modelos para realizar essa classificação.

A natureza desse problema exige que soluções sejam robustas a erros e errem o minimo de falso positivos possiveis.

## Modelos usados
Para realizar essa classificação foram testadas uma gama de modelos. De uma CNN a um modelo transformer para classificação(wav2vec)
A ideia é que esse algoritmos conseguiam atraves do audio bruto ou features extraidas como o Mel Spectograma, aprenderem a realizar a classificação.

* CNN
* SVM
* Regressão Logistica
* Resnet18 Feature Extractor
* Wav2vec2 like models - Fine Tuning - Feature Extractor

Entre algumas técnica que foram utilizadas para melhorar a perfomance dos modelos estas tiveram maior destaque:
* label smoothing
* data augmentation
* feature extraction
* transfer learning

O processo está mais bem documentato no relátorio final da competição: [Relatório_PAV.pdf](Relatório_PAV.pdf).

## Codigos

Os codigos usados estão implementados pasta a pasta em API que permitem a flexibilização de experimentos.

NN/training -> Codigos para treinar redes neurais e convolucionais.

ResNet18-Extractor - Codigos para usar a Resnet18 como feature extractor.

W2C - Codigos para treinar e testar modelos wav2vec2 like. 
## Modelo Final

O melhor modelo para a competição foi o fine tuning do modelo TODO , disponivel em TODO. Os outros modelos e técnicas exploradas tamém apresentaram bons resultados.

O modelo final foi testado no dataset ASVspoof 2019 Dataset que é um datasets ultilizados para avaliar essa task no estado da arte e coseguiu atingir boas métricas:

### Amostras do conjunto de avalição:

real : 7355

fake : 63882

### Métricas: 

Acuracia:0.9286

Precisão:0.9999

Recall:0.9205

F1-Score:0.9363

Equal Error Rate (EER):  0.0401
