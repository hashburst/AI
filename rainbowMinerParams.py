def get_mining_stats():
  with open("/path/to/rainbowminer/log.txt", "r") as log_file:
      # Log parsing per estrarre le performance
      accepted_shares = ...
      rejected_shares = ...
      gpu_temp = ...
      return accepted_shares, rejected_shares, gpu_temp

def update_miner_params(intensity, overclock):
    # Comando che aggiorna i parametri del miner
    command = f"./RainbowMiner --intensity {intensity} --overclock {overclock}"
    subprocess.run(command, shell=True)
