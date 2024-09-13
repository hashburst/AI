def update_miner_params(intensity, overclock):
    # Comando che aggiorna i parametri del miner
    command = f"./RainbowMiner --intensity {intensity} --overclock {overclock}"
    subprocess.run(command, shell=True)
