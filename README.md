# Processamento de Linguagem Natural

[Este repositório](https://github.com/ju-resplande/projeto_nlp) contém o projeto durante o curso de Processamento de Linguagem Natural, ofertado pelo Instituto de Informática da Universidade Federal de Goiás, ministrado pelo Prof. Dr. Arlindo Rodrigues Galvão Filho em 2022-2. 

O projeto está dividido em duas partes: `sentiment_analysis` e `recommender_system`.

## Instalação

### Dependências

- python 3
- conda. (Opcional para criação e gerenciamento de ambiente)

### Requerimentos

```
pip install -r requirements.txt
```


## Artefatos

Para baixar os artefatos necessários (conjunto de dados, modelos treinados, etc.), execute o comando:

```
python download_artifacts.py
```

## Análise de Sentimento
Versões disponíveis: 
* Baseline: Regressão logística com TF-IDF
* Bertimbau

Para reproduzir os experimentos, execute os notebooks `baseline.ipynb` e `bert.ipynb`


## Sistema de Recomendação

Versões disponíveis: 
* `sentiment_model`: without, bert_timbau
* `recommender_model`: svd, svdpp, nmf

- Rodar um treinamento:
    ```
    python -m luigi --module recommender_system.train TrainRS --sentiment-model{versão} --recommender-model {versão} --beta 0.1 --local-scheduler
    ```

Para reproduzir os experimentos
```
bash b2w.sh
```


## Notas

Desenvolvido em conjunto com os alunos: 
* Eduardo Augusto Santos Garcia
* Juliana Resplande Santanna Gomes
* Luana Guedes Barros Martins
* Werikcyano Lima Guimarães
