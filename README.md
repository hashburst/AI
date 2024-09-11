# AI
AI  per l'ottimizzazione del crypto-mining in un sistema di aggregazione di risorse distribuite ed orchestrate in cluster per il cloud computing 

Creare modelli di intelligenza artificiale che ottimizzino i parametri del software di crypto-mining richiede un'analisi del problema tale da individuare un framework open-source di machine learning adeguato alle risorse disponibili. Visto che si tratta di una configurazione avanzata con hardware di alto livello (HPE ProLiant con GPU NVidia H100), l'ottimizzazione dei parametri dei worker/miner può essere fatta utilizzando tecniche come il reinforcement learning o algoritmi di ottimizzazione bayesiana.

## 1. Requisiti:
Sistema operativo: Ubuntu 22.04/Proxmox VE
Hardware: HPE ProLiant x675d Gen 10 con 2 NVidia H100 PCIe
Software di Mining: RainbowMiner
Framework di AI: ti suggerisco di usare un framework open-source come PyTorch o TensorFlow che supportano CUDA per utilizzare al meglio le GPU NVidia H100.
Obiettivo: ottimizzare i parametri del miner per massimizzare le accepted share (100%) e ridurre a 0 le rejected share.

## 2. Scelta del Framework AI:
Considerando l'hardware NVidia H100 e la necessità di ottimizzazione basata su dati real-time, PyTorch è una scelta eccellente per la flessibilità e il supporto nativo di CUDA, ma TensorFlow potrebbe essere un'alternativa valida.

## 3. Approccio dell'Algoritmo:
Per l'algoritmo da implementare si intende usare un approccio basato su reinforcement learning (RL) o ottimizzazione bayesiana per migliorare progressivamente le performance del mining software.

L'algoritmo può monitorare i seguenti parametri per ogni worker:

- Accepted shares
- Rejected shares
- Temperature delle GPU
- Hashrate
- Efficienza energetica
- Altri parametri relativi ai singoli miner (es. intensità di lavoro, parallelizzazione)

Obiettivo: massimizzare le "accepted shares", minimizzare le "rejected shares" e mantenere la temperatura delle GPU sotto controllo.

## 4. Modello AI e RL Environment:

- Stato (State): i parametri del miner (es. intensità, parallelizzazione, hashrate corrente, temperatura GPU).

- Azione (Action): modifica dei parametri del miner, come intensità, tipo di algoritmo, impostazioni di overclocking, ecc.

- Ricompensa (Reward): la reward sarà massimizzata se le "accepted share" si attesteranno al 100% e "le rejected share" si approssimeranno allo 0%. Utilizzeremo la seguente funzione:

                          Reward = 100 − (RejectedShares) − (AcceptedShares<100)

In aggiunta a questa formula base, occorre introdurre una penalizzazione sulla reward se la temperatura delle GPU supera un certo limite. 

- Ambiente (Environment): il mining stesso può essere visto come un ambiente RL dove il modello esplora diverse configurazioni di parametri per ottimizzare i risultati.

A questo punto vediamo come ampliare la formula di Reward che consideri anche la temperatura delle GPU e come considerare il contributo proporzionale dell'utente rispetto al totale delle "accepted shares" nel Cluster. Introduciamo due nuovi concetti:

- Penalizzazione per alte temperature: se la temperatura della GPU supera il 90%, applichiamo una penalizzazione alla reward. La penalizzazione sarà più alta quanto maggiore è la temperatura rispetto alla soglia di 90°C.
  
- Proporzionalità del contributo alle accepted shares del cluster: il contributo di ogni worker/miner è proporzionale al numero di "accepted shares" che ha inviato rispetto al numero totale di "accepted shares" nel cluster di appartenenza.

### Estensione della formula

Definiamo le seguenti variabili aggiuntive:

- T_gpu: temperatura della GPU.
- T_max: temperatura massima tollerata (90°C in questo caso).
- accepted_user: numero di accepted shares inviate dal singolo miner.
- total_accepted_cluster: numero totale di accepted shares inviate dall'intero cluster.
- penalty_temp: penalizzazione basata sulla temperatura.
- share_contribution: contributo proporzionale di un miner in termini di accepted shares rispetto al cluster.

La nuova formula della reward può essere scritta come:

                          Reward = 100 − RejectedShares − Penalty(temp) − (AcceptedShares<100) × P(accept) × (1 − total_accepted_cluster/accepted_user)

Dove:

- Penalty(temp) =  max(0,T_gpu − T_max): penalizzazione basata sull'eccesso di temperatura rispetto alla soglia di 90°C.
- P(accept) = AcceptedShares<100: penalizzazione per non aver raggiunto il 100% di accepted shares.

### Implementazione del codice Python

Ecco come implementare la nuova formula di reward in Python:

                          def calculate_reward(rejected_shares, accepted_shares, accepted_user, total_accepted_cluster, T_gpu):
                              # Soglie e parametri
                              T_max = 90  # Temperatura massima tollerata in gradi Celsius
                              max_reward = 100  # Reward base
                              penalty_temp = max(0, T_gpu - T_max)  # Penalizzazione per temperatura
                          
                              # Penalizzazione se non si raggiunge il 100% di accepted shares
                              penalty_accepted = 100 - accepted_shares if accepted_shares < 100 else 0
                          
                              # Contributo proporzionale dell'utente nel cluster
                              if total_accepted_cluster > 0:
                                  share_contribution = 1 - (accepted_user / total_accepted_cluster)
                              else:
                                  share_contribution = 1  # Penalizzazione massima se non ci sono accepted shares nel cluster
                          
                              # Calcolo della reward finale
                              reward = max_reward - rejected_shares - penalty_temp - (penalty_accepted * share_contribution)
                          
                              # La reward non può essere negativa
                              return max(0, reward)
                          
                          # Esempio di utilizzo della funzione
                          rejected_shares = 10
                          accepted_shares = 95
                          accepted_user = 30
                          total_accepted_cluster = 100
                          T_gpu = 92
                          
                          reward = calculate_reward(rejected_shares, accepted_shares, accepted_user, total_accepted_cluster, T_gpu)
                          print(f"Reward calcolata: {reward}")

Spiegazione:

- penalty_temp: se la temperatura della GPU supera i 90°C, viene applicata una penalizzazione proporzionale all'eccesso di temperatura.
- penalty_accepted: penalizzazione se le accepted shares non raggiungono il 100%. Questa penalizzazione è moltiplicata per il contributo proporzionale del miner.
- share_contribution: determina quanto il contributo del miner sia importante rispetto al cluster, ovvero se il miner invia molte accepted shares, riceverà una penalizzazione minore, ma se contribuisce poco rispetto agli altri, la penalizzazione sarà maggiore.

Risultati attesi:

La reward finale sarà ridotta se:

- Il miner ha un numero di "rejected shares" alto.
- La temperatura della GPU è troppo alta.
- Le "accepted shares" sono basse rispetto al totale atteso (100%).
- Il contributo del miner rispetto al cluster è proporzionalmente basso.

Questo sistema incentiva i worker/miner a mantenere alta efficienza (accepted shares) e a mantenere le GPU a temperature adeguate per evitare surriscaldamenti, migliorando l'efficienza complessiva del mining.

## 5. Struttura del Codice:
Installazione delle dipendenze: prima di tutto, installare il necessario per far girare il modello AI su Ubuntu.

sudo apt update
sudo apt install python3-pip
pip3 install torch torchvision torchaudio tensorflow nvidia-pyindex nvidia-tensorflow
