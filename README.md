Claro! Aqui está o conteúdo do arquivo `README.md` sem os trechos de código:

---

# Preparação de Dados - FMA Large Dataset

Este projeto tem como objetivo a preparação dos dados do dataset FMA Large para treinamento de uma rede neural de classificação hierárquica multilabel. O dataset FMA Large contém informações detalhadas sobre músicas, como características de áudio e metadados, que serão utilizados para treinar um modelo que pode lidar com a complexidade das tarefas de classificação hierárquica.

## Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instruções de Uso](#instruções-de-uso)
- [Código](#código)
- [Resultados Esperados](#resultados-esperados)
- [Contribuição](#contribuição)
- [Licença](#licença)

## Visão Geral

O projeto foca na extração, transformação e organização dos dados contidos no FMA Large Dataset para que possam ser utilizados em um modelo de classificação hierárquica multilabel. A preparação dos dados inclui:

- Carregamento e limpeza dos dados.
- Transformação dos dados para o formato necessário para o modelo.
- Criação de labels hierárquicas para treinamento.
- Armazenamento dos dados preparados em um formato eficiente para uso durante o treinamento.

## Estrutura do Projeto

```
├── data/
│   ├── raw/                 # Dados brutos do FMA Large Dataset
│   ├── processed/           # Dados processados e prontos para uso
│   └── labels/              # Labels hierárquicos gerados para cada faixa
├── src/
│   ├── data_preparation.py  # Script principal para preparação dos dados
│   └── utils.py             # Funções auxiliares para transformação de dados
├── README.md                # Documentação do projeto
└── requirements.txt         # Dependências necessárias para o projeto
```

## Requisitos

Antes de iniciar, certifique-se de que você tem as seguintes dependências instaladas:

- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Instruções de Uso

1. **Download do Dataset:**
   Baixe o FMA Large Dataset no site oficial [FMA Large Dataset](https://github.com/mdeff/fma) e coloque os arquivos na pasta `data/raw/`.

2. **Preparação dos Dados:**
   Execute o script `data_preparation.py` para preparar os dados:

   ```bash
   python src/data_preparation.py
   ```

   Isso irá gerar os dados processados na pasta `data/processed/` e as labels na pasta `data/labels/`.

3. **Verificação dos Dados:**
   Certifique-se de que os dados foram processados corretamente, verificando os arquivos na pasta `data/processed/`.

## Resultados Esperados

Após a preparação dos dados, você deve ter:

- Um conjunto de dados processado e otimizado para treinamento.
- Labels hierárquicas para cada faixa de música, organizadas de acordo com a taxonomia musical.

Esses dados podem agora ser utilizados para treinar uma rede neural de classificação hierárquica multilabel.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir um issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

Esse é o conteúdo completo do arquivo `README.md` sem os códigos.
