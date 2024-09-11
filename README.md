# AI
AI  per l'ottimizzazione del crypto-mining in un sistema di aggregazione di risorse distribuite ed orchestrate in cluster per il cloud computing 

Creare modelli di intelligenza artificiale che ottimizzino i parametri del software di crypto-mining richiede un'analisi del problema tale da individuare un framework open-source di machine learning adeguato alle risorse disponibili. Visto che si tratta di una configurazione avanzata con hardware di alto livello (HPE ProLiant con GPU NVidia H100), l'ottimizzazione dei parametri dei worker/miner può essere fatta utilizzando tecniche come il reinforcement learning o algoritmi di ottimizzazione bayesiana.

** 1. Requisiti: **
Sistema operativo: Ubuntu 22.04/Proxmox VE
Hardware: HPE ProLiant x675d Gen 10 con 2 NVidia H100 PCIe
Software di Mining: RainbowMiner
Framework di AI: ti suggerisco di usare un framework open-source come PyTorch o TensorFlow che supportano CUDA per utilizzare al meglio le GPU NVidia H100.
Obiettivo: ottimizzare i parametri del miner per massimizzare le accepted share (100%) e ridurre a 0 le rejected share.

