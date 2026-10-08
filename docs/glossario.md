# Glossário do faisca

Este documento explica, em linguagem simples, os termos que aparecem nas
conversas, no README e no CLAUDE.md. Está em português porque é para o dono
do projecto ler. O resto da documentação do repositório fica em inglês.

Está organizado por temas. Cada termo tem uma explicação curta e, quando
ajuda, o que ele significa concretamente no faisca.

---

## 1. O que o programa faz

**Modelo (LLM)**
Um modelo de linguagem é um conjunto enorme de números (os pesos) mais uma
receita fixa de contas a fazer com eles. Dado um texto, as contas produzem,
para cada palavra possível, uma pontuação de "quão provável é ser a
próxima". O programa escolhe uma, acrescenta-a ao texto e repete. O faisca é
o programa que faz essas contas; o modelo é o ficheiro com os números.

**Pesos (weights)**
Os números aprendidos pelo modelo durante o treino. Um modelo de 360 M
(SmolLM2-360M) tem 360 milhões de pesos; um de 1.7 B tem 1.7 mil milhões.
Quase todo o tempo de execução é passado a ler pesos da memória e a
multiplicá-los.

**Camada (layer, block)**
O modelo está organizado em blocos iguais, repetidos em sequência. O
SmolLM2-360M tem 32 camadas, o 1.7B tem 24, o Llama 3.2 1B tem 16. Cada
camada tem duas partes: a atenção (ver secção 6) e a rede feed forward
(FFN). Quando dizemos "validado camada a camada" quer dizer que comparámos
a saída de cada camada com a do llama.cpp, não só o resultado final.

**Parâmetros do modelo (hparams)**
Os tamanhos: número de camadas, tamanho de cada vector interno (n_embd,
960 no 360M), número de cabeças de atenção (n_head), tamanho do
vocabulário (n_vocab, 49152 no SmolLM2), tamanho máximo de contexto. Vêm
no ficheiro GGUF.

**Arquitectura `llama`**
A receita de contas. SmolLM2 e Llama 3 usam a mesma receita (a do Llama),
por isso o faisca corre ambos. Outras famílias (Phi, Qwen, Mistral com
MoE) têm variações que ainda não estão implementadas.

---

## 2. Texto, tokens e velocidade

**Token**
A unidade com que o modelo trabalha. Não é uma letra nem uma palavra: é um
pedaço de texto, tipicamente 3 a 5 letras, às vezes uma palavra inteira com
o espaço à frente (" casa"), às vezes só um byte. O vocabulário é a lista
de todos os tokens que o modelo conhece. Cada token tem um número (id).

**Tokenizer**
A parte que converte texto em ids de tokens (encode) e ids em texto
(decode). O nosso é "byte level BPE": primeiro parte o texto em palavras
com regras fixas (o pre-tokenizer: `gpt-2`, `smollm`, `llama-bpe` são
conjuntos de regras diferentes), depois junta letras em tokens maiores
seguindo uma lista de fusões (merges) aprendida no treino. Foi validado
token a token contra o llama.cpp.

**BOS / EOS**
Tokens especiais de "início de texto" e "fim de texto". O modelo pára de
gerar quando produz EOS.

**Chat template**
A forma de embrulhar uma conversa em texto para o modelo: marcas do tipo
"aqui fala o utilizador", "aqui fala o assistente". O SmolLM2 usa o formato
ChatML (`<|im_start|>`), o Llama 3 usa o seu (`<|start_header_id|>`). O
faisca escolhe pelo vocabulário.

**Prompt**
O texto de entrada, já em tokens. Processar o prompt é uma fase diferente
de gerar: todos os tokens são conhecidos de antemão, por isso podem ser
tratados em lote (batch), e os pesos são lidos uma vez por lote em vez de
uma vez por token. É por isso que o prompt corre a centenas de tokens por
segundo e a geração a dezenas.

**Geração (generation, decode)**
Produzir tokens novos, um de cada vez. Cada token obriga a ler todos os
pesos do modelo outra vez. Com um modelo de 360 MB em Q8_0 e uma memória a
16 GB/s, isso dá no máximo cerca de 45 tokens por segundo, seja qual for
o processador. A geração é limitada pela memória, não pelas contas.

**tok/s**
Tokens por segundo. "33.8 / 98 tok/s" quer dizer 33.8 com uma thread e 98
com quatro. Nos benchmarks medimos sempre as duas fases: "prompt" (quantos
tokens de entrada por segundo) e "generation" (quantos tokens de saída).

**pp512, tg32**
Nomes do llama-bench: "prompt processing de 512 tokens" e "text generation
de 32 tokens". Usamos os mesmos tamanhos para comparar.

**Contexto (n_ctx, posições)**
Quantos tokens o modelo tem "à vista" ao mesmo tempo: o prompt mais o que
já gerou. Cada token ocupa uma posição (0, 1, 2, ...). O contexto máximo
é fixado ao arrancar (`-c` na CLI) e dimensiona o KV cache.

**Greedy, argmax**
Escolher sempre o token com maior pontuação. É determinístico: o mesmo
prompt dá sempre o mesmo texto. Usamos greedy em todos os testes
precisamente por isso: se dois programas estão certos, os tokens têm de
ser iguais. "Near tie" (quase empate) é quando os dois melhores tokens
têm pontuações quase iguais e qualquer arredondamento diferente troca a
escolha; não é um erro.

**Logits**
As pontuações de saída, uma por token do vocabulário, antes de serem
convertidas em probabilidades. Comparar logits é a forma mais exigente de
comparar dois runners.

**Sampling, temperatura**
Em vez de escolher sempre o melhor token, sortear entre os mais prováveis.
Dá texto mais variado. Ainda não está implementado no faisca; só greedy.

---

## 3. Como os números são guardados

**bit, byte**
Um byte são 8 bits. Um número pode ocupar 4 bytes (32 bits), 2 bytes (16
bits), ou, quando o comprimimos, meio byte (4 bits). Menos bits por peso
significa ficheiro mais pequeno e menos memória a ler por token, logo mais
rápido. Mas também menos precisão.

**float32, f32, FP32**
Número com vírgula flutuante em 32 bits. É a precisão "normal" de um
programa. Um modelo guardado assim é o mais fiel ao treino e o mais pesado
(360M em f32 = 1.4 GB). Usamo-lo como referência: o "caminho float32" do
faisca faz todas as contas em f32 e é comparado ao llama.cpp também em
f32, onde a diferença tem de ser quase zero.

**float16, f16, half**
O mesmo em 16 bits. Metade do espaço, cerca de 3 dígitos de precisão em
vez de 7. Nos ficheiros GGUF, F16 é um formato de pesos; no faisca
convertemos F16 para float32 ao carregar (ainda não há kernel f16). Também
usamos f16 para o KV cache (secção 6).

**bf16**
Outra variante de 16 bits, com menos precisão e mais alcance. Não usada
aqui.

**int8, int4, u8, s8**
Inteiros de 8 ou 4 bits. É nisto que os pesos quantizados são guardados.
"u8" é sem sinal (0 a 255), "s8" com sinal (-128 a 127).

**GGUF**
O formato de ficheiro do llama.cpp. Um ficheiro só com: os parâmetros do
modelo, o vocabulário do tokenizer, e todos os tensores de pesos, cada um
com o seu tipo (F32, F16, Q8_0, ...). O faisca lê-o directamente. O
nosso loader foi validado contra o leitor oficial em Python (gguf-py).

**Tensor**
Uma matriz de pesos com nome, por exemplo `blk.3.attn_q.weight` (a matriz
Q da atenção da camada 3). Um modelo tem centenas (290 no 360M).

**Memória vs disco, GB vs GiB**
Os pesos são lidos do disco uma vez e ficam em memória. "GB/s" nas notas
é a velocidade a que o processador consegue puxar pesos da memória RAM;
é o limite da geração. GiB (1024^3) e GB (1000^3) diferem 7%; nas tabelas
usamos o que a ferramenta deu.

---

## 4. Quantização: Q8_0, Q4_0, Q5_0, Q4_K, Q6_K, Q4_K_M

**Quantizar**
Comprimir os pesos: em vez de guardar cada peso com 32 bits, guardar um
inteiro pequeno (8, 5 ou 4 bits) e, por grupo de pesos, um factor de
escala. Peso ≈ inteiro × escala. Perde-se um pouco de precisão, ganha-se
2 a 8 vezes em tamanho e, como a geração é limitada pela memória, quase o
mesmo em velocidade.

**Bloco**
Os pesos são quantizados em grupos consecutivos que partilham a mesma
escala. Nos formatos "_0" o bloco tem 32 pesos; nos formatos "_K" tem 256
(um super bloco) dividido em sub blocos de 16 ou 32.

**Escala (d), mínimo (dmin)**
O factor por bloco. Guardado em f16 (formatos _0) ou f16 mais escalas de
6 bits por sub bloco (formatos _K). Alguns formatos guardam também um
mínimo por sub bloco (Q4_K), para que os inteiros sejam todos positivos.

**Os formatos que o faisca suporta**

| Nome | Bits por peso | Bloco | Como é guardado | Qualidade |
|------|---------------|-------|-----------------|-----------|
| F32  | 32  | nenhum | número exacto | referência |
| F16  | 16  | nenhum | número em 16 bits | quase igual a F32 |
| Q8_0 | 8.5 | 32 | 1 escala f16 + 32 inteiros de 8 bits | praticamente igual a F16 |
| Q5_0 | 5.5 | 32 | 1 escala f16 + 32 inteiros de 5 bits | boa |
| Q4_0 | 4.5 | 32 | 1 escala f16 + 32 inteiros de 4 bits | aceitável, perde-se algo |
| Q6_K | 6.56 | 256 | escala f16 + 16 escalas de 8 bits + inteiros de 6 bits | muito boa |
| Q4_K | 4.5 | 256 | escala e mínimo f16 + 8 escalas e 8 mínimos de 6 bits + inteiros de 4 bits | melhor que Q4_0 com o mesmo tamanho |

"Bits por peso" conta a escala: Q8_0 ocupa 34 bytes por 32 pesos, por
isso 8.5. O "_0" do nome vem do llama.cpp (a primeira variante do
formato; houve um Q4_1 com mínimo). O "_K" quer dizer "K-quants", a
segunda geração de formatos do llama.cpp, com blocos de 256 e escalas
dentro de escalas, que dá melhor qualidade para o mesmo tamanho.

**Q4_K_M, Q4_K_S, Q5_K_M, ...**
Isto não é um formato, é o nome de um ficheiro: uma mistura. O
llama-quantize não quantiza todas as matrizes com o mesmo formato; o
sufixo diz qual a receita de mistura. "Q4_K_M" (M de medium) usa Q4_K na
maioria, Q6_K nas matrizes mais sensíveis (as de saída e algumas da
atenção), e deixa o embedding e o output noutro formato. "S" (small) usa
menos Q6_K. Um ficheiro "Q4_K_M" do SmolLM2-360M é na verdade quase todo
Q5_0, porque os formatos _K precisam de matrizes com largura múltipla de
256 e o 360M tem 960 (960 / 256 não é inteiro); só as matrizes com largura
2560 (as do FFN) ficam em Q4_K ou Q6_K. O 1.7B tem larguras múltiplas de
256 e aí o ficheiro é mesmo Q4_K e Q6_K.

**Q4_1, Q5_1, Q2_K, Q3_K, Q5_K, IQ...**
Outros formatos do llama.cpp que o faisca não suporta. Se um ficheiro os
tiver, o loader recusa ou converte para float32 (ver "Conversão").

**Dequantizar**
Fazer a operação inversa: inteiro × escala, obtendo float32. O faisca pode
dequantizar tudo ao carregar (`--dequantize` / `--f32`) para correr o
modelo em float32 puro. É lento (5 tok/s) e serve só para validação.

**Conversão ao carregar**
Quando o processador não tem o kernel para um tipo (por exemplo Q6_K num
CPU sem VNNI), o loader converte essa matriz para float32 na memória. O
ficheiro não muda. Fica correcto, mas gasta 4 a 8 vezes mais memória e é
mais lento.

**Packed, packing, empacotado**
Ao carregar, o faisca reorganiza os pesos quantizados na memória, de 16
em 16 linhas entrelaçadas, para que as instruções vectoriais (secção 5)
leiam 16 linhas de uma vez. Os bytes são os mesmos, só a ordem muda. É
isto que demora no arranque (0.4 s no Llama 3.2 1B com 4 threads, 10 s
no 1.7B quando a memória está fragmentada).

**Activações, activações quantizadas (Q8_0 / Q8_K)**
Activação é o vector que entra em cada matriz (a saída da camada
anterior). Para multiplicar inteiros por inteiros (o que o VNNI faz),
também a activação tem de ser quantizada, no momento, para 8 bits. Usamos
exactamente o mesmo esquema do llama.cpp: Q8_0 (blocos de 32) para
matrizes Q8_0 / Q4_0 / Q5_0 e Q8_K (blocos de 256) para Q4_K / Q6_K. É
por isso que os nossos resultados quantizados são iguais aos do
llama.cpp bit a bit nas matrizes, e por isso que o Vulkan (que não
quantiza activações) fica mais perto do float32.

**Ruído de quantização, "inherent noise"**
Um modelo Q8_0 não dá os mesmos logits que o mesmo modelo em float32: dá
uns 4% de diferença (4.4e-2). Isso não é um erro nosso; o llama.cpp tem a
mesma diferença. O teste correcto é: o faisca em Q8_0 contra o llama.cpp
em Q8_0 com as mesmas entradas tem de dar zero, e os tokens gerados têm
de ser os mesmos.

---

## 5. O processador: threads, vectores, VNNI, AMX

**Thread**
Um fio de execução. Com 4 threads o trabalho de cada token é dividido por
4 núcleos. "1 / 4 threads" nas tabelas são os dois casos medidos. O faisca
divide o trabalho de forma que o resultado seja bit a bit igual com 1, 2 ou
4 threads (cada grupo de linhas é feito inteiro por uma thread).

**Thread pool**
Um conjunto de threads criadas uma vez ao arrancar e reutilizadas. Criar
threads por tarefa seria lento. "Fork/join" é cada vez que se distribui
uma tarefa e se espera que todas acabem; custa alguns microsegundos, e com
modelos pequenos (o draft de 135M) isso pesa.

**Kernel**
Uma função pequena e muito optimizada que faz uma operação pesada: por
exemplo "multiplica esta matriz Q4_0 por este vector". Há um kernel por
tipo de peso e por conjunto de instruções do processador.

**matvec, matmul, GEMM, GEMV**
Matriz vezes vector (um token) ou matriz vezes matriz (vários tokens do
prompt de uma vez). GEMV e GEMM são os nomes clássicos. Na geração é
sempre matvec; no prompt é matmul em lotes de 32 tokens.

**MAC, GMAC**
Multiply-accumulate: uma multiplicação mais uma soma, a operação básica.
GMAC = mil milhões delas. "131 GMAC/s" é o tecto do VNNI a 2.1 GHz.

**SIMD, instruções vectoriais, AVX2, AVX-512**
Instruções que fazem a mesma conta a vários números de uma vez. AVX2
trabalha com registos de 256 bits (chamados ymm: 32 inteiros de 8 bits de
cada vez); AVX-512 com 512 bits (zmm: 64 de cada vez). Nem todos os
processadores têm AVX-512 (os Intel de portátil recentes e os AMD antes
do Zen 4 não têm).

**VNNI**
Vector Neural Network Instructions: uma instrução (`vpdpbusd`) que
multiplica 4 pares de inteiros de 8 bits e soma o resultado, para 16 ou
32 posições ao mesmo tempo. É a instrução certa para pesos de 8 bits
contra activações de 8 bits. "AVX-512 VNNI" é a versão de 512 bits;
"AVX-VNNI" é a mesma instrução nos registos de 256 bits, para processadores
sem AVX-512. Os kernels do faisca existem nas duas versões; a de 256 bits
para Q5_0 e K-quants é o trabalho em curso.

**AMX**
Advanced Matrix Extensions: unidades de matriz dos Xeon recentes
(Sapphire Rapids). Fazem multiplicações de blocos 16×64 numa instrução. O
llama.cpp usa-as para o prompt. Medimos e vimos que só ganhava 1.25x na
parte que podia acelerar (10% do tempo do prompt), porque o que resta
depois (aplicar escalas em float) domina; por isso ficou de lado.

**F16C**
Instruções para converter entre f16 e float32 rapidamente. Usadas para
ler o KV cache em f16.

**Scalar, kernel de referência**
Versão sem instruções vectoriais, um número de cada vez, escrita da forma
mais óbvia. É lenta mas serve de referência: os kernels vectoriais têm de
dar o mesmo resultado bit a bit. É também o que corre em processadores
sem AVX.

**Cache (L1, L2, L3), memória**
Memórias pequenas e rápidas dentro do processador. L3 nesta máquina tem
260 MiB, grande o suficiente para um modelo de 135M caber inteiro, o que
distorce medições (os pesos nunca vêm da RAM). Por isso os benchmarks de
streaming usam 400 MiB.

**Bound (memory bound, compute bound)**
"Limitado por". Memory bound: a velocidade é ditada por quantos bytes se
conseguem ler por segundo; mais núcleos ou instruções melhores não
ajudam. Compute bound: ditada pelas contas; aqui sim ajudam. A geração é
memory bound, o prompt é compute bound.

**Prefetch (prefetcht0)**
Uma instrução que diz ao processador "vai buscar esta linha de memória,
vou precisar dela daqui a pouco". Permite ter mais leituras em curso ao
mesmo tempo. Foi o que levou o Q4_0 de 7.9 para 13.8 GB/s.

**Huge pages**
Normalmente a memória é gerida em páginas de 4 KiB; huge pages são de 2
MiB. Com pesos de centenas de MB, menos páginas significa menos trabalho
para o processador a traduzir endereços: +25% no kernel. O preço é que
o sistema tem de encontrar blocos contíguos de 2 MiB (compactação), e é
isso que faz o carregamento variar de 0.5 s a 10 s.

**ulp**
Unit in the last place: a menor diferença representável entre dois
floats. "1 ulp de expf" quer dizer que a nossa exponencial dá o valor
exacto ou o vizinho imediato.

**-O2, -O3, -ffp-contract**
Níveis de optimização do compilador. Medimos a -O3 porque a -O2 o
compilador guarda valores intermédios em memória (spills) e as medições
em cache ficam 3x erradas. `-ffp-contract=off` impede o compilador de
fundir multiplicação e soma por conta própria, para que Debug e Release
dêem resultados idênticos.

---

## 6. Dentro de uma camada: atenção, KV cache, rope

**Embedding**
A primeira operação: o id do token é uma linha numa tabela; essa linha
(960 números no 360M) é o vector com que a camada 1 começa. É o único
sítio onde se lê uma linha só, em vez de uma matriz inteira ("gather").

**rms_norm**
Normalizar um vector (dividir pelo seu tamanho) e multiplicar por pesos
aprendidos. Feita antes da atenção e antes do FFN.

**Atenção (attention), cabeças (heads), Q K V**
A parte onde cada token "olha" para os tokens anteriores. Para cada token
calculam-se três vectores: Q (query, "o que procuro"), K (key, "o que
ofereço") e V (value, "o que entrego"). A pontuação de cada token anterior
é Q·K; passa por softmax (vira pesos que somam 1); a saída é a soma dos V
pesados por esses pesos. Isto é feito em paralelo por várias "cabeças"
(15 no 360M), cada uma com a sua fatia do vector (head_dim = 64).
`n_head_kv` menor que `n_head` (GQA) quer dizer que várias cabeças Q
partilham os mesmos K e V, para poupar memória.

**Softmax, exp**
Converter pontuações em pesos positivos que somam 1, através da
exponencial. Escrevemos a nossa própria exponencial vectorizada porque
`std::exp` chamada 82 mil vezes por token estava a custar 0.37 s por
prompt.

**KV cache**
Os vectores K e V de cada token, de cada camada, guardados para não os
recalcular quando chega o token seguinte. É o que faz a geração custar
"um token" em vez de "todo o contexto". Ocupa memória proporcional ao
contexto: 160 MiB para 2048 posições no 360M em float32.

**KV cache em f16 (kv_f16, --kv-f32)**
Guardar o K e o V em 16 bits em vez de 32: metade da memória e mais
rápido com contextos longos (21.5 contra 20.4 tok/s a 1045 posições).
Perde-se um pouco de precisão (1.9e-3 contra float32, igual ao llama.cpp
com a mesma opção). É o default; `--kv-f32` desliga.

**Tiles, key_index**
As chaves são guardadas em blocos de 16 posições transpostos, para que
uma instrução vectorial compare uma query com 16 chaves de uma vez. É um
pormenor de layout que explica porque um snapshot de 19 tokens ocupa
tanto como 32.

**Flash attention**
Forma de calcular a atenção sem guardar a matriz de pontuações inteira.
O llama.cpp usa-a por omissão; o nosso reference tool desliga-a para a
comparação ser exacta. O faisca tem a sua versão em tiles.

**RoPE, rope, rope scaling, rope factors**
Rotary Position Embedding: a forma de dizer ao modelo em que posição está
cada token, rodando os vectores Q e K por um ângulo que depende da
posição. `freq_base` (100000 no SmolLM2, 500000 no Llama 3) e `n_rot`
são os seus parâmetros. "Rope scaling" é um ajuste para o modelo aceitar
contextos mais longos do que foi treinado; o Llama 3 traz uma tabela de
factores (`rope_freqs.weight`) que divide as frequências. Foi o que
faltava para carregar ficheiros Llama 3.

**FFN, SwiGLU, gate / up / down**
A segunda metade da camada: três matrizes. `gate` e `up` multiplicam o
vector; a saída é `silu(gate) * up` (SwiGLU é o nome dessa função) e
`down` traz o resultado ao tamanho original. É onde estão 2/3 dos pesos.

**Residual (add)**
Somar a saída de cada parte à sua entrada, para que a informação passe de
camada em camada sem se perder. O llama.cpp chama "ffn_out" a duas coisas
diferentes; é uma das armadilhas apontadas no CLAUDE.md.

**Output, lm_head**
A última matriz: transforma o vector final em logits, um por token do
vocabulário. No SmolLM2 é a mesma tabela do embedding (tied). É a matriz
maior do 1B (5.5 ms por token no Llama 3.2).

---

## 7. Técnicas de aceleração ao nível do runner

**Batch, lote (Runner::BATCH = 32)**
Processar 32 tokens do prompt de uma vez. Os pesos são lidos uma vez por
lote, por isso o prompt é 5 a 6 vezes mais rápido do que a geração por
token.

**Speculative decoding, draft, `--draft`**
Usar um modelo pequeno (o draft, SmolLM2-135M) para adivinhar os próximos
3 tokens e o modelo grande (o target) para os confirmar todos numa só
passagem. Os tokens confirmados saem de uma vez. Quando o draft acerta,
o grande leu os seus pesos uma vez por 3 ou 4 tokens em vez de uma vez
por token. O texto é exactamente igual ao do modelo grande sozinho. Ganho
medido: 1.3 a 1.4x com uma thread, pouco com quatro, porque o draft custa
0.3 a 0.4 de um token do target.

**Session, snapshot, prefix reuse, `--session`**
Guardar o estado da conversa (os tokens e o KV cache) para não repetir o
prompt. Se o novo pedido começa pelos mesmos tokens que o anterior
(prefixo comum), só os tokens novos são processados. `snapshot / restore`
guarda e repõe na memória; `save / load` escreve em disco (ficheiro
"FSCA"). O chat da CLI usa isto para continuar de onde ficou.

**Offset table**
No ficheiro de sessão, uma tabela no início que diz onde está cada
pedaço (tokens, chaves da camada 3, valores da camada 3, ...). Permite
ler um pedaço sem ler o resto.

---

## 8. GPU: Vulkan

**GPU**
A placa gráfica. Tem muito mais largura de banda de memória do que o CPU
(centenas de GB/s contra 16), por isso em princípio gera tokens muito
mais depressa. O preço é que cada operação tem de ser escrita como um
programa pequeno que corre lá (shader), e os dados têm de ser copiados
para a memória da placa.

**Vulkan, CUDA, OpenGL**
Três APIs para falar com a GPU. CUDA é só NVIDIA; OpenGL é antiga e feita
para gráficos; Vulkan é moderna, corre em NVIDIA, AMD, Intel e no
llvmpipe, e tem compute shaders. Foi a escolha do dono.

**Compute shader, GLSL, SPIR-V**
Um shader é o programa que corre na GPU. Escrevemo-los em GLSL
(linguagem parecida com C) e compilamos para SPIR-V (o formato binário
que o Vulkan aceita). Os binários ficam embebidos no executável
(`shaders_spv.cpp`, gerado pelo `tools/embed_spirv.py`), para não haver
ficheiros a carregar em runtime.

**Workgroup, subgroup, dispatch**
Workgroup: o conjunto de threads da GPU que trabalha em conjunto numa
tarefa (no nosso matvec, uma linha da matriz por workgroup). Subgroup: o
subconjunto que executa em lockstep (8 no llvmpipe, 32 na NVIDIA, 64 na
AMD). Dispatch: mandar executar um shader com N workgroups.

**Command buffer, barrier**
Lista de dispatches gravada e submetida de uma vez. Uma barreira entre
dois dispatches garante que o segundo vê os resultados do primeiro.

**Staging buffer**
Memória intermédia visível ao CPU, por onde os dados passam a caminho da
memória da placa.

**llvmpipe**
Uma implementação de Vulkan que corre no CPU (Mesa). É o que existe nesta
máquina. Serve para verificar que o código está certo, não para medir
velocidade. Os números de velocidade da GPU têm de vir da máquina do
dono.

**Loader aberto em runtime (dlopen / LoadLibrary)**
O faisca não liga directamente à biblioteca do Vulkan; abre-a ao arrancar
se existir. Assim o mesmo executável corre em máquinas sem Vulkan, só
sem a opção `--vulkan`.

---

## 9. Validação: como sabemos que está certo

**Referência (llama.cpp)**
O programa de referência da comunidade, com o qual comparamos tudo. Não o
usamos no faisca, só nos testes: `llama_layer_dump` corre o mesmo modelo
no llama.cpp e grava a saída de cada tensor; `compare_layers.py` compara
com a nossa.

**rel L2, 4.5e-6**
Diferença relativa entre dois vectores: a raiz da soma dos quadrados das
diferenças, dividida pela norma da referência. "4.5e-6" é 0.00045%:
diferenças de arredondamento. "4.4e-2" é 4.4%: ruído de quantização.
"1.8e-7" para Q5_0 é o arredondamento de um produto em float, não um
erro de kernel.

**Bit identical, bit a bit**
Os bytes de saída são exactamente iguais, não apenas próximos. É o que
exigimos entre kernels vectoriais e scalar, entre 1 e 4 threads, entre
Debug e Release, e entre uma sessão retomada e uma corrida de uma vez.

**Dump, runner_dump**
Gravar em ficheiro a saída de cada tensor de cada camada para uma dada
entrada. `cmp` de dois dumps é o teste de "bit identical".

**matmul_check**
Testa cada kernel de matriz isoladamente com as entradas reais que o
llama.cpp usou (gravadas no dump). "Zero error em 193 matrizes" quer
dizer que para Q8_0 e Q4_0 damos os mesmos bytes que o llama.cpp.

**ASan, UBSan, TSan**
Sanitizers: versões do executável instrumentadas pelo compilador para
apanhar acessos a memória inválida (ASan), comportamento indefinido como
overflows (UBSan) e corridas entre threads (TSan). "ASan clean" quer dizer
que correu sem apanhar nada. Correm mais devagar, por isso são builds à
parte (`build-sanitize`, `build-tsan`).

**ctest, run_all.sh**
`ctest` corre os testes unitários do build; `tests/run_all.sh` faz o
release mais os sanitizers.

**Unit test (ops_unit, tokenizer_unit)**
Testes pequenos de cada função com entradas construídas: o kernel
vectorial dá o mesmo que o scalar? o decode de um token devolve o texto
certo? "183 checks, 0 failed".

**Benchmark, bench/**
Medições de velocidade reproduzíveis, com os resultados em
`bench/README.md`. Regra do projecto: só se escreve uma estrutura de
dados própria quando um benchmark em dados reais prova que é mais rápida.

**Scratchpad, scratch bench**
Experiências descartáveis fora do repositório (bench_amx, bench_q5,
runner_phases). Os resultados ficam nas notas; o código não é commitado.

---

## 9b. Voz: Whisper e os termos de áudio

**Amostra, taxa de amostragem, 16 kHz**
Um ficheiro de som é uma lista de números (amostras), cada um a altura da
onda num instante. 16 kHz quer dizer 16000 amostras por segundo. O Whisper
foi treinado com 16 kHz mono (um canal); áudio a 44.1 kHz ou estéreo tem
de ser convertido antes (o `ffmpeg` faz isso; o nosso leitor de WAV recusa
outras taxas e tira a média dos canais).

**WAV, PCM 16 bits**
O formato de ficheiro mais simples: um cabeçalho e as amostras cruas, cada
uma um inteiro de 16 bits. Sem compressão. MP3 e OGG são comprimidos e
precisariam de um descodificador nosso.

**FFT (transformada de Fourier)**
A operação que converte um pedaço de som (400 amostras, 25 ms) na lista
de frequências que o compõem e com que força. Fazemos uma a cada 10 ms
(160 amostras de salto).

**Janela de Hann**
Antes da FFT, o pedaço de 400 amostras é multiplicado por uma curva que
vale zero nas pontas e um no meio, para o corte não inventar frequências
falsas.

**Espectrograma mel, "mel"**
A imagem que o modelo realmente ouve: para cada 10 ms, 80 valores que
dizem quanta energia há em 80 bandas de frequência espaçadas como o
ouvido humano as separa (escala mel). Os valores são logaritmos,
cortados a 8 abaixo do máximo e normalizados. 30 segundos dão 3000
colunas por 80 linhas. O nosso é igual bit a bit ao do whisper.cpp.

**Encoder, decoder**
O Whisper tem duas metades. O encoder lê o espectrograma inteiro (3000
colunas, reduzidas a 1500 por uma convolução de passo 2) e produz 1500
vectores que resumem o som. O decoder é um modelo de linguagem como os
SmolLM2: gera o texto token a token, mas em cada camada, além de olhar
para os tokens anteriores, olha para os 1500 vectores do encoder.

**Atenção cruzada (cross attention)**
Esse olhar do decoder para o encoder. As chaves e os valores vêm do
encoder (calculados uma vez por janela e guardados), as queries vêm do
texto.

**Convolução 1D**
A primeira operação do encoder: cada saída mistura três colunas vizinhas
do espectrograma com pesos aprendidos. Duas delas, a segunda a saltar de
duas em duas colunas.

**Layer norm, GELU**
As variantes que o Whisper usa em vez do rms_norm e do SwiGLU: a layer
norm subtrai a média e divide pelo desvio, com peso e bias; o GELU é a
activação, uma curva suave parecida com um ReLU.

**Tokens especiais do Whisper**
O prompt do decoder é fixo: início (sot), língua (`<|pt|>`), tarefa
(transcrever ou traduzir) e "sem timestamps". O modelo responde com
tokens de texto até ao fim de texto (eot). Os tokens de timestamp marcam
tempos de 20 em 20 ms; ainda não os usamos.

**Janela de 30 segundos**
O encoder processa sempre 30 s de cada vez (o que falta é preenchido com
silêncio). Áudio mais longo é cortado em janelas sucessivas; o whisper.cpp
usa os timestamps para cortar nas pausas, nós por agora cortamos aos 30 s.

**WER (word error rate)**
A medida de qualidade da transcrição: palavras trocadas, a mais ou a
menos, a dividir pelo total. 5% quer dizer uma palavra errada em vinte.

**tiny, base, small, medium, large**
Os tamanhos do Whisper: 39 M, 74 M, 244 M, 769 M e 1550 M de pesos. Os
ficheiros `ggml-*.bin` do whisper.cpp guardam-nos em f16 (metade do
tamanho em bytes). O faisca converte para float32 ao carregar e, por
omissão, quantiza as matrizes para Q8_0 com os kernels que já tinha.

**Tempo real, "6x real time"**
Quantos segundos de áudio se transcrevem por segundo de relógio. 6x quer
dizer que um minuto de gravação demora dez segundos. Abaixo de 1x é mais
lento do que ouvir.

---

## 10. Os ficheiros em /home/user/models

| Ficheiro | O que é |
|----------|---------|
| smollm2-360m-instruct-q8_0.gguf | o modelo alvo, oficial, pesos em Q8_0 |
| smollm2-360m-instruct-f32.gguf | o mesmo, dequantizado para float32 (referência) |
| smollm2-360m-instruct-q4_0.gguf | feito com llama-quantize a partir do f32 |
| smollm2-360m-instruct-q4_k_m.gguf | idem, mistura Q4_K_M (quase tudo Q5_0, ver secção 4) |
| draft/smollm2-135m-instruct-*.gguf | o modelo pequeno para speculative decoding |
| 1.7b/smollm2-1.7b-instruct-q4_k_m.gguf | modelo maior, oficial, Q4_K e Q6_K a sério |
| llama32/llama-3.2-1b-instruct-q8_0.gguf | Llama 3.2 1B, com rope factors e tokenizer llama-bpe |
| whisper/ggml-{tiny,base,small}.bin | Whisper multilingue do whisper.cpp, f16 |
| whisper/ggml-*-f32.bin | os mesmos em float32, feitos com tests/make_f32_whisper.py (referência) |

Os ficheiros `*-f32.gguf` existem só para validação: os dois lados
(faisca e llama.cpp) lêem os mesmos números exactos e qualquer diferença
é erro de implementação, não de quantização.

---

## 11. Leitura rápida de uma frase típica

"Q4_K_M, 4 threads, 118 tok/s de geração contra 54 do llama.cpp, 2.3e-6
do float32 no Vulkan, 16 tokens idênticos."

Quer dizer: com o ficheiro misturado Q4_K_M, usando 4 núcleos, o faisca
produz 118 tokens por segundo onde o llama.cpp produz 54; a versão GPU
tem uma diferença de arredondamento (0.0002%) em relação à execução em
float32; e os 16 primeiros tokens gerados são exactamente os mesmos que
os do llama.cpp, por isso o resultado está certo.
