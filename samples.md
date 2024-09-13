# Esempio generazione file di configurazione

                  api_key = "YOUR_API_KEY"
                  miner = "miner-software"
                  algo = "sha256, scrypt, x11, ..."
                  pool = "stratum.pool.tld"
                  port = "3333"
                  account = "account_name"
                  subaccount = "worker_name"
                  coin = "ALT Coin List"
                  num_miners = 5
                  
                  generate_config(api_key, miner, algo, pool, port, account, subaccount, coin, num_miners)

# Esempio di utilizzo della funzione

                  rejected_shares = 10
                  accepted_shares = 95
                  accepted_user = 30
                  total_accepted_cluster = 100
                  T_gpu = 92
                  
                  reward = calculate_reward(rejected_shares, accepted_shares, accepted_user, total_accepted_cluster, T_gpu)
                  print(f"Reward calcolata: {reward}")
