# Tradução Automática e o Conjunto de Dados Tatoeba

Este projeto implementa a seção 9.5 do Dive into Deep Learning sobre tradução automática usando o conjunto de dados Tatoeba para tradução inglês-francês.

## Implementação Completa

### Código no Notebook (`Traducao_Automatica_Ana_Goes.ipynb`)

✅ **Download e pré-processamento do conjunto de dados Tatoeba**
- Download direto do dataset oficial: `https://www.manythings.org/anki/fra-eng.zip`
- 237.838 pares de frases inglês-francês disponíveis
- Pré-processamento com normalização de texto e tokenização em nível de palavra

✅ **Tokenização e obtenção do vocabulário**
- Tokenização por espaços (word-level)
- Tokens especiais: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Frequência mínima de 2 para inclusão no vocabulário
- Vocabulários separados para inglês e francês

✅ **Modelo Seq2Seq completo**
- Encoder-Decoder com GRU
- Embeddings independentes por idioma
- Teacher forcing durante treinamento
- Decodificação greedy para inferência

## Respostas às Perguntas da Seção 9.5.7

### 1. Tente valores diferentes do argumento `num_examples` na função `load_data_nmt`. Como isso afeta os tamanhos do vocabulário do idioma de origem e do idioma de destino?

**Resposta baseada nos resultados experimentais do notebook:**

O parâmetro `num_examples` tem um impacto **direto e significativo** nos tamanhos dos vocabulários de origem e destino. Conforme demonstrado no notebook através dos experimentos realizados:

**Resultados Experimentais Obtidos:**
```
num_examples=   600 |V_en|=   359 |V_fr|=   390
num_examples= 12000 |V_en|=  2744 |V_fr|=  3552
```

**Padrões Observados:**

1. **Relação Direta**: À medida que `num_examples` aumenta de 600 para 12.000, os tamanhos dos vocabulários aumentam drasticamente:
   - Vocabulário inglês: 359 → 2.744 tokens
   - Vocabulário francês: 390 → 3.552 tokens

2. **Crescimento Não-Linear**: O crescimento é exponencial/logarítmico, não linear, indicando que mais exemplos introduzem palavras cada vez mais raras

3. **Vocabulário Francês Maior**: O vocabulário de destino (francês) consistentemente apresenta tamanho maior que o vocabulário de origem (inglês) para os mesmos valores de `num_examples`

**Justificativa Técnica:**
- **Mais exemplos = mais diversidade lexical**: Cada novo exemplo pode introduzir palavras únicas ao vocabulário
- **Frequência mínima (min_freq=2)**: Limita o crescimento, mas permite expansão gradual conforme mais dados são processados
- **Morfologia do francês**: Maior complexidade morfológica (conjugações, gêneros, acentos, contrações) resulta em vocabulário mais rico que o inglês
- **Lei de Zipf**: A distribuição de frequências de palavras segue uma lei de potência, onde poucas palavras são muito frequentes e muitas são raras

### 2. O texto em alguns idiomas, como chinês e japonês, não tem indicadores de limite de palavras (por exemplo, espaço). A tokenização em nível de palavra ainda é uma boa ideia para esses casos? Por que ou por que não?

**Resposta com justificativas fundamentadas:**

**NÃO, a tokenização em nível de palavra NÃO é adequada para idiomas sem delimitadores de palavra** pelos seguintes motivos fundamentais:

#### **Problemas Fundamentais:**

1. **Ausência de Delimitadores Naturais**
   - Como mencionado na pergunta, chinês e japonês não usam espaços para separar palavras
   - A segmentação em palavras torna-se uma tarefa de NLP complexa por si só
   - Requer ferramentas especializadas de segmentação (ex: Jieba para chinês, MeCab para japonês)

2. **Ambigüidade de Segmentação**
   - Uma sequência de caracteres pode ser segmentada de múltiplas formas válidas
   - Erros de segmentação propagam-se para todo o pipeline de tradução

3. **Explosão do Vocabulário**
   - Vocabulários extremamente grandes (centenas de milhares de tokens)
   - Problema de tokens raros (OOV - Out of Vocabulary) exacerbado
   - Dificuldade de treinamento e inferência devido ao tamanho do vocabulário

#### **Alternativas Mais Eficazes:**

1. **Tokenização em Nível de Caractere**
   - **Vantagens**: Vocabulário limitado (~3000-8000 caracteres), sem ambiguidade de segmentação
   - **Desvantagens**: Perda de informações morfológicas, sequências muito longas

2. **Tokenização em Nível de Subpalavra**
   - **BPE (Byte Pair Encoding)**: Usado em modelos como GPT, BERT
   - **WordPiece**: Usado em modelos como BERT
   - **SentencePiece**: Solução unificada para múltiplos idiomas
   - **Vantagens**: Balanceia granularidade e eficiência, trata palavras raras efetivamente

3. **Tokenização Morfológica**
   - Segmentação em unidades morfológicas
   - Especialmente útil para idiomas com morfologia rica

#### **Referências e Evidências:**

Esta análise é consistente com:

1. **Seção 9.5 do Dive into Deep Learning**: *"Usando tokenização em nível de palavra, o tamanho do vocabulário será significativamente maior do que usando tokenização em nível de caractere. Para aliviar isso, podemos tratar tokens raros como o mesmo token desconhecido."*

2. **Literatura Científica**: 
   - [Wu et al., 2016](https://arxiv.org/abs/1604.00788) demonstra a eficácia de subpalavras para NMT
   - [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) introduz BPE para NMT
   - [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226) apresenta SentencePiece

3. **Prática Industrial**: Modelos modernos como GPT, BERT, T5 usam tokenização em subpalavras para idiomas sem espaços

4. **Evidência Experimental**: No nosso próprio experimento, mesmo com inglês e francês (idiomas com espaços), vimos que o vocabulário francês cresceu mais rapidamente, indicando que idiomas com morfologia mais rica se beneficiam menos da tokenização em nível de palavra

**Conclusão**: Para idiomas sem delimitadores de palavra, a tokenização em nível de subpalavra é a abordagem mais eficaz, oferecendo o melhor equilíbrio entre granularidade semântica e eficiência computacional.

## Resultados do Modelo

### Treinamento
- **Épocas**: 5
- **Loss inicial**: 3.96
- **Loss final**: 1.24
- **Melhoria**: Redução de perplexidade

### Exemplos de Tradução
```
i love you ! -> je vous en prie .
this is a test . -> c ' est une histoire .
where is the bathroom ? -> où se trouve la gare ?
```

## Estrutura do Projeto

- **`Traducao_Automatica_Ana_Goes.ipynb`**: Notebook principal com implementação completa da seção 9.5
- **`README.md`**: Este arquivo com respostas às perguntas e documentação

## Características da Implementação

✅ **Dataset Tatoeba real** - 237.838 pares inglês-francês  
✅ **Modelo Seq2Seq completo** - Encoder-Decoder com GRU  
✅ **Experimentos funcionais** - responde às perguntas dos exercícios  
✅ **Treinamento completo** - 5 épocas com curvas de loss/perplexidade  
✅ **Inferência implementada** - decodificação greedy para tradução  
✅ **Análise de vocabulário** - impacto do num_examples demonstrado  
