import json
import os

# Path alla directory BIN di RainbowMiner
CONFIG_PATH = "/path/to/RainbowMiner/BIN"

# Funzione per aggiornare i parametri del miner in tempo reale
def update_miner_config(miner_name, new_params):
    config_file = os.path.join(CONFIG_PATH, f"{miner_name}.config.json")  # Presupponendo formato JSON
    try:
        # Carica la configurazione corrente
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Aggiorna i parametri nel file
        for key, value in new_params.items():
            if key in config_data:
                config_data[key] = value

        # Scrivi le modifiche al file di configurazione
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)

        print(f"Parametri del miner {miner_name} aggiornati con successo.")
    except Exception as e:
        print(f"Errore nell'aggiornamento del file di configurazione del miner {miner_name}: {str(e)}")

# Esempio di utilizzo della funzione
miner_name = "miner_example"
new_params = {
    "gpu_threads": 2,
    "intensity": 30,
    "worksize": 256
}
update_miner_config(miner_name, new_params)
